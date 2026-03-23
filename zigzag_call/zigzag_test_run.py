import socket
import struct
import json
import pandas as pd

def send_request(host, port, params):
    try:
        # 1. 创建 Socket 连接
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            print(f"Connected to {host}:{port}")

            # 2. 准备 JSON 数据
            json_payload = json.dumps(params).encode('utf-8')
            
            # 3. 封装 Header (4字节长度，大端序)
            header = struct.pack('!I', len(json_payload))
            
            # 4. 发送数据
            sock.sendall(header + json_payload)
            print(f"Sent: {params}")

            # 5. 接收响应 Header
            resp_header = sock.recv(4)
            if not resp_header:
                return None
            
            resp_length = struct.unpack('!I', resp_header)[0]
            
            # 6. 接收响应 Body
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
    # 测试数据
    m, k, n = 5120, 4096, 4096
    archs = ["ws", "os"]
    raw_results = []

    print(f"📊 硬件架构倍数对比实验 | 规模: {m}x{k}x{n}")
    print("-" * 60)

    for arch in archs:
        print(f"正在获取 {arch.upper()} 数据...", end="", flush=True)
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
        
        # 以第一个架构为基准 (Base)
        base = df.iloc[0]
        
        # 计算对比数据
        comparison = []
        for i in range(len(df)):
            current = df.iloc[i]
            comparison.append({
                "指标维度": ["Latency (延迟)", "Energy (能量)", "EDP (综合)"],
                "Arch": current['Arch'],
                "绝对数值": [
                    f"{current['Latency']:.2e}", 
                    f"{current['Energy']:.2e}", 
                    f"{current['EDP']:.2e}"
                ],
                "对比倍数": [
                    f"{current['Latency'] / base['Latency']:.4f}x",
                    f"{current['Energy'] / base['Energy']:.4f}x",
                    f"{current['EDP'] / base['EDP']:.4f}x"
                ]
            })

        # 打印精美对比表
        print("\n" + "="*75)
        print(f"{'性能指标对比清单 (以 ' + base['Arch'] + ' 为基准 1.0000x)':^75}")
        print("="*75)
        
        for item in comparison:
            print(f"--- 架构: {item['Arch']} ---")
            temp_df = pd.DataFrame({
                "指标维度": item['指标维度'],
                "绝对数值": item['绝对数值'],
                "相对于基准的倍数": item['对比倍数']
            })
            print(temp_df.to_string(index=False))
            print("-" * 75)

        # 深度分析逻辑
        os_idx = df.index[df['Arch'] == 'OS'].tolist()
        ws_idx = df.index[df['Arch'] == 'WS'].tolist()
        
        if os_idx and ws_idx:
            os_edp = df.loc[os_idx[0], 'EDP']
            ws_edp = df.loc[ws_idx[0], 'EDP']
            ratio = ws_edp / os_edp
            winner = "OS" if ratio > 1 else "WS"
            print(f"💡 深度洞察: {winner} 架构的综合能效(EDP)比另一方优越了 {abs(1-ratio)*100:.2f}%")