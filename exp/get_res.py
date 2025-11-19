import os
import csv
import sys
import pandas as pd
import numpy as np
import json

chunked_groups={
    'sharegpt':4,
    'govreport':5,
    'cnndm':1
}

def read_search_res(csv_path):
    """读取search_res.csv文件并返回latency和energy的值"""
    try:
        df=pd.read_csv(csv_path)
        latency=df['latency'].mean()
        energy=df['energy'].mean()
        latency = np.exp(np.log(df['latency']).mean())
        energy = np.exp(np.log(df['energy']).mean())
        latency=df['latency'].mean()
        energy=df['energy'].mean()
        return latency, energy
    except Exception as e:
        print(f"读取文件 {csv_path} 时出错: {e}", file=sys.stderr)
        return None, None

def process_folder_a(folder_a_path, print_all=False):
    """
    处理单个A文件夹，找出latency*energy最小的B文件夹
    
    参数:
        folder_a_path: A文件夹的路径
        print_all: 如果为True，打印所有B文件夹的结果（按乘积从小到大排序）
    
    返回:
        min_folder_b: 乘积最小的B文件夹名称
        min_product: 最小乘积值
        found_any_csv: 是否找到任何CSV文件
    """
    min_product = float('inf')
    min_latency=float('inf')
    min_energy=float('inf')

    min_folder_b = None
    found_any_csv = False
    results = []  # 存储所有结果用于排序
    
    # 遍历A文件夹下的所有子文件夹
    try:
        for folder_b in os.listdir(folder_a_path):
            folder_b_path = os.path.join(folder_a_path, folder_b)
            
            # 确保是文件夹
            if not os.path.isdir(folder_b_path):
                continue
            
            # 检查是否存在search_res.csv
            csv_path = os.path.join(folder_b_path, 'exec_res.csv')
            if not os.path.exists(csv_path):
                continue
            
            found_any_csv = True
            
            # 读取数据并计算乘积
            latency, energy = read_search_res(csv_path)
            if latency is not None and energy is not None:
                product = latency * energy
                results.append((folder_b, product, latency, energy))
                
                # 更新最小值
                if product < min_product:
                    min_latency=latency
                    min_energy=energy
                    min_product = product
                    min_folder_b = folder_b
    
    except Exception as e:
        print(f"处理文件夹 {folder_a_path} 时出错: {e}", file=sys.stderr)
        return None, None, False
    
    # 如果需要打印所有结果，按乘积从小到大排序
    if print_all and results:
        results.sort(key=lambda x: x[1])  # 按乘积排序
        folder_a_name = os.path.basename(folder_a_path)
        print(f"\n  {folder_a_name} 的所有子文件夹结果（按乘积从小到大）:")
        print(f"  {'-' * 70}")
        for i, (folder_b, product, latency, energy) in enumerate(results, 1):
            print(f"  {i}. {folder_b}:")
            print(f"     latency*energy = {product:.2e}")
            print(f"     (latency={latency:.2e}, energy={energy:.2e})")
        print(f"  {'-' * 70}")
    
    return min_folder_b, min_product, found_any_csv, min_latency, min_energy

def main():
    assert len(sys.argv)==2, "Usage: python3 get_res.py <seq length distri>"
    seq_length_distri=sys.argv[1]

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取同级目录下的所有文件夹
    folder_list = []
    for item in os.listdir(script_dir):
        item_path = os.path.join(script_dir, item)
        # 只处理文件夹，排除脚本文件本身
        if os.path.isdir(item_path):
            folder_list.append(item)
    
    # 按名称排序
    folder_list.sort()
    
    if not folder_list:
        print("未找到任何文件夹")
        return
    
    # 处理每个文件夹
    print("=" * 60)
    print("分析结果:")
    print("=" * 60)
    
    res_map={}

    for folder_name in folder_list:
        folder_path = os.path.join(script_dir, folder_name)
        min_folder_b, min_product, found_any_csv, min_latency, min_energy = process_folder_a(folder_path)
        
        if min_folder_b:
            # print(f"{folder_name}/{min_folder_b} (latency*energy = {min_product:.2e})")
            print(f"'{folder_name}/{min_folder_b}',{min_latency*min_energy:.2e},(latency={min_latency:.2e}, energy={min_energy:.2e})")
            res_map[folder_name]=(min_latency,min_energy)
        elif found_any_csv:
            print(f"{folder_name}: 找到CSV文件但无法读取有效数据")
        else:
            print(f"{folder_name}: 未找到包含search_res.csv的子文件夹")
    
    print("=" * 60)
    print("汇报每一个数据集下不同硬件的chunked prefill执行的结果：")
    # dataset=['sharegpt','cnndm']
    dataset=[seq_length_distri]
    hardware=['os','ws','he'] # ,'hecross','hecb'
    res_for_scenario={}
    for d in dataset:
        for h in hardware:
            key=f"chunked_prefill_{d}_{h}"

            latency,energy=res_map[key]
            total_latency=latency*chunked_groups[d]
            total_energy=energy*chunked_groups[d]
            total_edp=total_latency*total_energy
            print(f"{d} {h}          total edp:{total_edp:.2e} (latency:{total_latency:.2e}, energy:{total_energy:.2e})")
            res_key=f"chunked_prefill_{d}_{h}"
            res_for_scenario[res_key]=total_edp
            
    print("汇报每一个数据集下不同硬件的vLLM混合执行的结果：")
    for d in dataset:
        for h in hardware:
            prefill_key=f"prefill_{d}_{h}"
            decode_key=f"decode_{d}_{h}"

            prefill_latency,prefill_energy=res_map[prefill_key]
            decode_latency,decode_energy=res_map[decode_key]
            total_latency=prefill_latency+decode_latency*chunked_groups[d]
            total_energy=prefill_energy+decode_energy*chunked_groups[d]
            total_edp=total_latency*total_energy
            print(f"{d} {h}          total edp:{total_edp:.2e} (latency:{total_latency:.2e}, energy:{total_energy:.2e})")
            res_key=f"vllm_{d}_{h}"
            res_for_scenario[res_key]=total_edp

    print("汇报每一个数据集下不同硬件的Orca混合执行的结果：")
    for d in dataset:
        for h in hardware:
            prefill_key=f"mixed_{d}_{h}"
            decode_key=f"decode_{d}_{h}"

            prefill_latency,prefill_energy=res_map[prefill_key]
            decode_latency,decode_energy=res_map[decode_key]
            total_latency=prefill_latency+decode_latency*(chunked_groups[d]-1)
            total_energy=prefill_energy+decode_energy*(chunked_groups[d]-1)
            total_edp=total_latency*total_energy
            print(f"{d} {h} total edp:{total_edp:.2e} (latency:{total_latency:.2e}, energy:{total_energy:.2e})")
            res_key=f"orca_{d}_{h}"
            res_for_scenario[res_key]=total_edp

    print("汇报不同硬件跨数据集的运行结果")
    scenario=['chunked_prefill','vllm','orca']
    res_print={}
    for s in scenario:
        for h in hardware:
            edp_mul=1.0
            cnt=0
            for d in dataset:
                cnt+=1
                res_key=f"{s}_{d}_{h}"
                edp=res_for_scenario[res_key]
                edp_mul*=edp
            edp_geo=edp_mul**(1.0/cnt)
            print(f"{s} {h} geo mean edp:{edp_geo:.2e}")
            res_print[f"{s}_{h}_compass"]=edp_geo
    print(json.dumps(res_print, indent=4))
    # print("汇报每一个数据集下不同硬件的vLLM混合执行的结果：")
    # dataset=['sharegpt']
    # hardware=['os','ws','heside','hecross','hecb']
    # for d in dataset:
    #     for h in hardware:
    #         prefill_key=f"prefill_{d}_{h}"
    #         decode_key=f"decode_{d}_{h}"

    #         prefill_latency,prefill_energy=res_map[prefill_key]
    #         decode_latency,decode_energy=res_map[decode_key]
    #         total_latency=prefill_latency/4+decode_latency*5
    #         total_energy=prefill_energy/4+decode_energy*5
    #         total_edp=total_latency*total_energy
    #         print(f"{d} {h} total edp:{total_edp:.2e} (latency:{total_latency:.2e}, energy:{total_energy:.2e})")



if __name__ == "__main__":
    main()