import os
import json
import csv

def process_data():
    # 获取当前脚本所在目录
    base_dir = "."
    final_result = {}

    # 遍历当前目录下的所有项
    for item in os.listdir(base_dir):
        # 仅处理文件夹，跳过文件（例如脚本本身或生成的输出文件）
        if not os.path.isdir(item) or item.startswith('.'):
            continue
        
        folder_path = item
        
        try:
            # ==========================================
            # 1. 处理延迟 (Latency)
            # ==========================================
            csv_path = os.path.join(folder_path, "hetero_phase", "exec_out", "exec_res_hetero.csv")
            latencies = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    latencies.append(float(row['latency']))
            
            # 计算平均值并除以 1e9
            if latencies:
                latency_avg = (sum(latencies) / len(latencies)) / 1e9
            else:
                latency_avg = 0.0

            # ==========================================
            # 2. 处理能量 (Energy List)
            # ==========================================
            energy_path = os.path.join(folder_path, "exec_energy_detail.json")
            with open(energy_path, 'r', encoding='utf-8') as f:
                energy_data = json.load(f)
            
            chiplet_energy_raw = 0.0
            dram_energy_raw = 0.0
            nop_energy_raw = 0.0

            # 遍历所有 core
            for core, entries in energy_data.items():
                for entry in entries:
                    chiplet_energy_raw += entry.get("calcEnergy", 0) + entry.get("ubufEnergy", 0)
                    dram_energy_raw += entry.get("dramEnergy", 0)
                    nop_energy_raw += entry.get("nocEnergy", 0)
            
            # 整理为列表并除以 1e12
            energy_list = [
                chiplet_energy_raw / 1e12,
                dram_energy_raw / 1e12,
                nop_energy_raw / 1e12
            ]

            # ==========================================
            # 3. 处理货币成本 (Monetary Cost List)
            # ==========================================
            cost_path = os.path.join(folder_path, "exec_mc_detail.json")
            with open(cost_path, 'r', encoding='utf-8') as f:
                cost_data = json.load(f)
            
            # 按顺序提取 cost_compute, cost_IO, cost_os_overall
            cost_list = [
                cost_data.get("cost_compute", 0),
                cost_data.get("cost_IO", 0),
                cost_data.get("cost_os_overall", 0)
            ]

            # ==========================================
            # 4. 计算乘积 (Product)
            # ==========================================
            total_energy = sum(energy_list)
            total_cost = sum(cost_list)
            product = latency_avg * total_energy * total_cost

            # ==========================================
            # 5. 保存结果
            # ==========================================
            final_result[item] = [
                latency_avg,
                energy_list,
                cost_list,
                product
            ]
            
            print(f"成功处理文件夹: {item}")

        except FileNotFoundError as e:
            print(f"跳过 {item}: 缺少必要的文件 ({e.filename})")
        except KeyError as e:
            print(f"跳过 {item}: 数据格式不匹配，缺少键值 {e}")
        except Exception as e:
            print(f"跳过 {item}: 发生未知错误 - {e}")

    # 将结果输出到 JSON 文件
    output_filename = "aggregated_results.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("{\n")
        
        lines = []
        for key, value in final_result.items():
            # json.dumps 默认不会添加多余的缩进和换行，非常适合将内层列表紧凑化
            # ensure_ascii=False 防止中文文件夹名称出现乱码
            value_str = json.dumps(value, ensure_ascii=False)
            # 拼接成 JSON 键值对格式，外层加 4 个空格缩进
            lines.append(f'    "{key}": {value_str}')
            
        # 用逗号和换行符连接所有项
        f.write(",\n".join(lines))
        f.write("\n}")
    
    print(f"\n所有数据处理完毕！结果已保存至 {output_filename}")

if __name__ == "__main__":
    process_data()