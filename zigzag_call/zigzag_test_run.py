import socket
import struct
import json
import pandas as pd

def send_request(host, port, params):
    try:
        # 1. Create Socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            print(f"Connected to {host}:{port}")

            # 2. Prepare JSON data
            json_payload = json.dumps(params).encode('utf-8')
            
            # 3. Encapsulate Header (4-byte length, big-endian)
            header = struct.pack('!I', len(json_payload))
            
            # 4. Send data
            sock.sendall(header + json_payload)
            print(f"Sent: {params}")

            # 5. Receive response Header
            resp_header = sock.recv(4)
            if not resp_header:
                return None
            
            resp_length = struct.unpack('!I', resp_header)[0]
            
            # 6. Receive response Body
            resp_body = b''
            while len(resp_body) < resp_length:
                chunk = sock.recv(resp_length - len(resp_body))
                if not chunk:
                    break
                resp_body += chunk
            
            return json.loads(resp_body.decode('utf-8'))

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Test data
    m, k, n = 5120, 4096, 4096
    archs = ["ws", "os"]
    raw_results = []

    print(f"Hardware Architecture Multiplier Comparison Experiment | Scale: {m}x{k}x{n}")
    print("-" * 60)

    for arch in archs:
        print(f"Fetching {arch.upper()} data...", end="", flush=True)
        test_params = {
            "m": m, "k": k, "n": n,
            "arch": arch,
            "mac": [4, 4, 8, 8],
            "buf": 2048*1024
        }
        resp = send_request("127.0.0.1", 18888, test_params)
        if resp:
            raw_results.append({
                "Arch": arch.upper(),
                "Latency": resp['t'],
                "Energy": resp['e'],
                "EDP": resp['t'] * resp['e']
            })
            print(" Done")
        else:
            print(" Fail")

    if len(raw_results) >= 2:
        df = pd.DataFrame(raw_results)
        
        # Use the first architecture as the baseline (Base)
        base = df.iloc[0]
        
        # Calculate comparison data
        comparison = []
        for i in range(len(df)):
            current = df.iloc[i]
            comparison.append({
                "Metric": ["Latency", "Energy", "EDP"],
                "Arch": current['Arch'],
                "Absolute Value": [
                    f"{current['Latency']:.2e}", 
                    f"{current['Energy']:.2e}", 
                    f"{current['EDP']:.2e}"
                ],
                "Comparison Ratio": [
                    f"{current['Latency'] / base['Latency']:.4f}x",
                    f"{current['Energy'] / base['Energy']:.4f}x",
                    f"{current['EDP'] / base['EDP']:.4f}x"
                ]
            })

        # Print formatted comparison table
        print("\n" + "="*75)
        print(f"{'Performance Metrics Comparison (Baseline: ' + base['Arch'] + ' at 1.0000x)':^75}")
        print("="*75)
        
        for item in comparison:
            print(f"--- Architecture: {item['Arch']} ---")
            temp_df = pd.DataFrame({
                "Metric": item['Metric'],
                "Absolute Value": item['Absolute Value'],
                "Ratio to Baseline": item['Comparison Ratio']
            })
            print(temp_df.to_string(index=False))
            print("-" * 75)

        # In-depth analysis logic
        os_idx = df.index[df['Arch'] == 'OS'].tolist()
        ws_idx = df.index[df['Arch'] == 'WS'].tolist()
        
        if os_idx and ws_idx:
            os_edp = df.loc[os_idx[0], 'EDP']
            ws_edp = df.loc[ws_idx[0], 'EDP']
            ratio = ws_edp / os_edp
            winner = "OS" if ratio > 1 else "WS"
            print(f"Insight: The comprehensive energy efficiency (EDP) of the {winner} architecture is {abs(1-ratio)*100:.2f}% better than the other.")