import argparse
import csv
import json
import socket
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


GPT3_7B = {
    "d_model": 4096,
    "n_head": 32,
    "d_head": 128,
    "d_ffn": 16384,
}

DEFAULT_SEQ_LENGTHS = [128, 1024, 5120, 10240]
DEFAULT_MAC = [4, 4, 8, 8]
DEFAULT_BUFFER_BYTES = 2048 * 1024
STAGE_ORDER = ["QKV Gen", "QK", "FFN1", "FFN2"]


def send_request(host: str, port: int, params: Dict[str, object], timeout: float = 600.0) -> Dict[str, float]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect((host, port))

        payload = json.dumps(params).encode("utf-8")
        sock.sendall(struct.pack("!I", len(payload)) + payload)

        header = _recv_exact(sock, 4)
        response_length = struct.unpack("!I", header)[0]
        response_body = _recv_exact(sock, response_length)
        response = json.loads(response_body.decode("utf-8"))
        if not isinstance(response, dict) or "t" not in response or "e" not in response:
            raise ValueError(f"Unexpected ZigZag response for {params}: {response}")
        return response


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    chunks = []
    received = 0
    while received < num_bytes:
        chunk = sock.recv(num_bytes - received)
        if not chunk:
            raise ConnectionError(f"Socket closed after {received}/{num_bytes} bytes.")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def parse_int_list(value: str) -> List[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    try:
        return [int(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def gpt3_7b_stage_shapes(seq_len: int, qk_mode: str) -> List[Tuple[str, int, int, int]]:
    d_model = GPT3_7B["d_model"]
    d_ffn = GPT3_7B["d_ffn"]
    d_head = GPT3_7B["d_head"]
    n_head = GPT3_7B["n_head"]

    if qk_mode == "per_head":
        qk_m = seq_len
    elif qk_mode == "aggregate_heads":
        qk_m = seq_len * n_head
    else:
        raise ValueError(f"Unsupported qk_mode={qk_mode!r}")

    return [
        ("QKV Gen", seq_len, d_model, 3 * d_model),
        ("QK", qk_m, d_head, seq_len),
        ("FFN1", seq_len, d_model, d_ffn),
        ("FFN2", seq_len, d_ffn, d_model),
    ]


def run_case(
    host: str,
    port: int,
    stage: str,
    seq_len: int,
    m: int,
    k: int,
    n: int,
    arch: str,
    mac: List[int],
    buffer_bytes: int,
    timeout: float,
) -> Dict[str, object]:
    params = {
        "m": m,
        "k": k,
        "n": n,
        "arch": arch,
        "mac": mac,
        "buf": buffer_bytes,
    }
    response = send_request(host, port, params, timeout=timeout)
    latency = float(response["t"])
    energy = float(response["e"])
    return {
        "stage": stage,
        "seq_len": seq_len,
        "m": m,
        "k": k,
        "n": n,
        "arch": arch.upper(),
        "latency": latency,
        "energy": energy,
        "edp": latency * energy,
    }


def collect_results(
    host: str,
    port: int,
    seq_lengths: Iterable[int],
    qk_mode: str,
    mac: List[int],
    buffer_bytes: int,
    timeout: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seq_len in seq_lengths:
        for stage, m, k, n in gpt3_7b_stage_shapes(seq_len, qk_mode):
            for arch in ["os", "ws"]:
                print(
                    f"Running {stage:7s} seq={seq_len:<5d} "
                    f"shape={m}x{k}x{n} arch={arch.upper()}",
                    flush=True,
                )
                rows.append(
                    run_case(
                        host=host,
                        port=port,
                        stage=stage,
                        seq_len=seq_len,
                        m=m,
                        k=k,
                        n=n,
                        arch=arch,
                        mac=mac,
                        buffer_bytes=buffer_bytes,
                        timeout=timeout,
                    )
                )
    return rows


def build_summary_rows(raw_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_case: Dict[Tuple[str, int, int, int, int], Dict[str, Dict[str, object]]] = {}
    for row in raw_rows:
        key = (
            str(row["stage"]),
            int(row["seq_len"]),
            int(row["m"]),
            int(row["k"]),
            int(row["n"]),
        )
        by_case.setdefault(key, {})[str(row["arch"])] = row

    summary_rows: List[Dict[str, object]] = []
    stage_rank = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
    for key in sorted(by_case, key=lambda item: (item[1], stage_rank.get(item[0], len(stage_rank)))):
        stage, seq_len, m, k, n = key
        case = by_case[key]
        os_row = case.get("OS")
        ws_row = case.get("WS")
        if os_row is None or ws_row is None:
            continue

        os_edp = float(os_row["edp"])
        ws_edp = float(ws_row["edp"])
        winner = "OS" if os_edp < ws_edp else "WS"
        best = min(os_edp, ws_edp)
        worst = max(os_edp, ws_edp)
        summary_rows.append(
            {
                "stage": stage,
                "seq_len": seq_len,
                "shape": f"{m}x{k}x{n}",
                "os_edp": os_edp,
                "ws_edp": ws_edp,
                "ws_over_os": ws_edp / os_edp if os_edp else float("inf"),
                "winner": winner,
                "winner_gain_pct": (worst / best - 1.0) * 100.0 if best else 0.0,
            }
        )
    return summary_rows


def format_edp(value: float) -> str:
    return f"{value:.2e}"


def format_os_ws_ratio(row: Dict[str, object]) -> str:
    ws_edp = float(row["ws_edp"])
    if ws_edp == 0:
        return "inf"
    return f"{float(row['os_edp']) / ws_edp:.2f}"


def build_os_ws_matrix(summary_rows: List[Dict[str, object]]) -> Tuple[List[str], List[List[str]]]:
    seq_lengths = sorted({int(row["seq_len"]) for row in summary_rows})
    row_by_stage_seq = {
        (str(row["stage"]), int(row["seq_len"])): row
        for row in summary_rows
    }

    stages = [stage for stage in STAGE_ORDER if any(str(row["stage"]) == stage for row in summary_rows)]
    extra_stages = sorted({str(row["stage"]) for row in summary_rows} - set(stages))
    stages.extend(extra_stages)

    headers = ["SeqLen"] + stages
    table: List[List[str]] = []
    for seq_len in seq_lengths:
        row = [str(seq_len)]
        for stage in stages:
            summary_row = row_by_stage_seq.get((stage, seq_len))
            row.append(format_os_ws_ratio(summary_row) if summary_row else "-")
        table.append(row)
    return headers, table


def print_summary_table(summary_rows: List[Dict[str, object]]) -> None:
    headers, table = build_os_ws_matrix(summary_rows)
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table)) if table else len(headers[i])
        for i in range(len(headers))
    ]
    line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep = "-+-".join("-" * width for width in widths)
    print("\nGPT3-7B OS/WS EDP comparison")
    print("Each cell is OS_EDP / WS_EDP. Lower than 1.00 means OS has lower EDP.")
    print(line)
    print(sep)
    for row in table:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers, table = build_os_ws_matrix(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare OS/WS EDP for GPT3-7B QKV Gen, QK, FFN1, and FFN2 "
            "matrix multiplications at different sequence lengths."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument("--seq-lengths", type=parse_int_list, default=DEFAULT_SEQ_LENGTHS)
    parser.add_argument(
        "--qk-mode",
        choices=["per_head", "aggregate_heads"],
        default="per_head",
        help=(
            "per_head uses QK shape S x d_head x S. "
            "aggregate_heads uses (S*n_head) x d_head x S."
        ),
    )
    parser.add_argument("--mac", type=parse_int_list, default=DEFAULT_MAC)
    parser.add_argument("--buffer-bytes", type=int, default=DEFAULT_BUFFER_BYTES)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/gpt3_7b_os_ws_edp.csv"),
        help="CSV path for the 2D OS_EDP/WS_EDP table.",
    )
    args = parser.parse_args()

    raw_rows = collect_results(
        host=args.host,
        port=args.port,
        seq_lengths=args.seq_lengths,
        qk_mode=args.qk_mode,
        mac=args.mac,
        buffer_bytes=args.buffer_bytes,
        timeout=args.timeout,
    )
    summary_rows = build_summary_rows(raw_rows)
    print_summary_table(summary_rows)
    write_csv(args.csv, summary_rows)
    print(f"\nSaved summary CSV to {args.csv}")


if __name__ == "__main__":
    main()
