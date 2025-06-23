from hyperopt import fmin, tpe, hp, Trials
import json
import subprocess
import os
import csv
from functools import partial
import pickle
import numpy as np
import copy
from pathlib import Path

from BO_params import chiplet_count_options, chiplet_type_list, buffer_size_list, compute_unit_list, nop_bw_options, dram_bw_options, decode_micro_batch_options, prefill_micro_batch_options

micro_batch_options = prefill_micro_batch_options

# 目录设置
now_d = Path(__file__).resolve().parent
base_directory = now_d / "exp_diff/Carch_Cmapping_decode_hybrid/"
homo_directory = base_directory / "homo_phase/"
hetero_directory = base_directory / "hetero_phase/"


# 优化参数
init_rounds = 50
homo_rounds = init_rounds + 100  # 同构优化轮数
hetero_rounds = 50  # 异构优化轮数
total_rounds = homo_rounds + hetero_rounds

seed = 42
rstate = np.random.default_rng(seed)

mc_limit = 49.14685

def cost_func(latency, energy, mc):
    return latency * energy * (max(mc, mc_limit) / mc_limit)

algo = partial(tpe.suggest, gamma=0.5, n_startup_jobs=init_rounds)

# 确保所有目录存在
for phase_dir in [homo_directory, hetero_directory]:
    dirs = [
        phase_dir, 
        phase_dir / "hardware_params/", 
        phase_dir / "search_out/", 
        phase_dir / "search_log/", 
        phase_dir / "exec_out/"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# ======================= 同构优化部分 =======================
def build_homo_space():
    return {
        "chiplet_count_options": hp.choice("chiplet_count_options", list(range(len(chiplet_count_options)))),
        "chiplet_type": hp.choice("chiplet_type", list(range(len(chiplet_type_list)))),
        "nop_bw": hp.choice("nop_bw", list(range(len(nop_bw_options)))),
        "dram_bw": hp.choice("dram_bw", list(range(len(dram_bw_options)))),
        "micro_batch": hp.choice("micro_batch", list(range(len(micro_batch_options)))),
        "buffer": hp.choice("buffer", list(range(len(buffer_size_list)))),
        "compute": hp.choice("compute", list(range(len(compute_unit_list)))),
    }

def create_homo_objective(directory):
    first_flag = True
    
    def objective(params):
        num_chiplets = chiplet_count_options[params["chiplet_count_options"]]
        chiplet_type = chiplet_type_list[params["chiplet_type"]]
        buffer = buffer_size_list[params["buffer"]]
        compute = compute_unit_list[params["compute"]]
        
        config_json = {
            "num_chiplets": num_chiplets,
            "nop_bw": nop_bw_options[params["nop_bw"]],
            "dram_bw": dram_bw_options[params["dram_bw"]],
            "micro_batch": micro_batch_options[params["micro_batch"]],
            "chiplets": []
        }
        
        for _ in range(num_chiplets):
            config_json["chiplets"].append({
                "type": chiplet_type,
                "buffer_size": buffer,
                "compute_units": compute
            })
        
        global log_id
        json_path = directory / f"hardware_params/input_{log_id}.json"
        csv_path = directory / f"search_out/output_{log_id}.csv"
        compass_out_path = directory / f"search_log/compass_{log_id}.out"
        run_cmd = now_d / "build/compass"
        compass_config_path = base_directory / "compass_config_search.json"
        res_csv_path = directory / "homo_search_results.csv"
        log_id += 1
        
        try:
            with open(json_path, "w") as f:
                json.dump(config_json, f, indent=2)
            
            with open(compass_out_path, "w") as outfile:
                subprocess.run([run_cmd, compass_config_path, json_path, csv_path], 
                              check=True, stdout=outfile, stderr=outfile, cwd=base_directory)
            
            with open(csv_path, "r") as f:
                header = f.readline()
                values = f.readline().strip().split(",")
                latency, energy, mc = map(float, values[:3])
            
            total_cost = cost_func(latency, energy, mc)
            
            nonlocal first_flag
            if first_flag:
                if os.path.exists(res_csv_path):
                    os.remove(res_csv_path)
                with open(res_csv_path, "w", newline="") as log_file:
                    writer = csv.writer(log_file)
                    writer.writerow(["latency", "energy", "mc", "total_cost"])
                first_flag = False
            
            with open(res_csv_path, "a", newline="") as log_file:
                writer = csv.writer(log_file)
                writer.writerow([latency, energy, mc, total_cost])
            
            return total_cost
        except Exception as e:
            print(f"评估失败: {e}")
            return float("inf")
    
    return objective

# ======================= 异构优化部分 =======================
def build_hetero_space():
    chiplet_configs = []
    space = {
        "config_index": hp.choice("config_index", list(range(len(chiplet_count_options)))),
        "nop_bw": hp.choice("nop_bw", list(range(len(nop_bw_options)))),
        "dram_bw": hp.choice("dram_bw", list(range(len(dram_bw_options)))),
        "micro_batch": hp.choice("micro_batch", list(range(len(micro_batch_options))))
    }

    for count in chiplet_count_options:
        chiplets = []
        for i in range(count):
            suffix = f"_n{count}_{i}"
            space[f"type{suffix}"] = hp.choice(f"type{suffix}", list(range(len(chiplet_type_list))))
            space[f"buffer{suffix}"] = hp.choice(f"buffer{suffix}", list(range(len(buffer_size_list))))
            space[f"compute{suffix}"] = hp.choice(f"compute{suffix}", list(range(len(compute_unit_list))))
            chiplets.append({
                "type": f"type{suffix}",
                "buffer": f"buffer{suffix}",
                "compute": f"compute{suffix}"
            })
        chiplet_configs.append({
            "num_chiplets": count,
            "chiplets": chiplets
        })
    return space, chiplet_configs

def create_hetero_objective(chiplet_configs, directory):
    first_flag = True
    
    def objective(params):
        config_index = params["config_index"]
        config = chiplet_configs[config_index]
        chiplets = config["chiplets"]
        num_chiplets = config["num_chiplets"]
        
        config_json = {
            "num_chiplets": num_chiplets,
            "nop_bw": nop_bw_options[params["nop_bw"]],
            "dram_bw": dram_bw_options[params["dram_bw"]],
            "micro_batch": micro_batch_options[params["micro_batch"]],
            "chiplets": []
        }

        for i in range(num_chiplets):
            chiplet = chiplets[i]
            type_idx = params.get(chiplet["type"])
            buffer_idx = params.get(chiplet["buffer"])
            compute_idx = params.get(chiplet["compute"])

            if type_idx is None or buffer_idx is None or compute_idx is None:
                continue

            config_json["chiplets"].append({
                "type": chiplet_type_list[type_idx],
                "buffer_size": buffer_size_list[buffer_idx],
                "compute_units": compute_unit_list[compute_idx]
            })
        
        global log_id
        json_path = directory / f"hardware_params/input_{log_id}.json"
        csv_path = directory / f"search_out/output_{log_id}.csv"
        compass_out_path = directory / f"search_log/compass_{log_id}.out"
        run_cmd = now_d/"build/compass"
        compass_config_path = base_directory / "compass_config_search.json"
        res_csv_path = directory / "hetero_search_results.csv"
        log_id += 1
        
        try:
            with open(json_path, "w") as f:
                json.dump(config_json, f, indent=2)
            
            with open(compass_out_path, "w") as outfile:
                subprocess.run([run_cmd, compass_config_path, json_path, csv_path], 
                              check=True, stdout=outfile, stderr=outfile, cwd=base_directory)
            
            with open(csv_path, "r") as f:
                header = f.readline()
                values = f.readline().strip().split(",")
                latency, energy, mc = map(float, values[:3])
            
            total_cost = cost_func(latency, energy, mc)
            
            nonlocal first_flag
            if first_flag:
                if os.path.exists(res_csv_path):
                    os.remove(res_csv_path)
                with open(res_csv_path, "w", newline="") as log_file:
                    writer = csv.writer(log_file)
                    writer.writerow(["latency", "energy", "mc", "total_cost"])
                first_flag = False
            
            with open(res_csv_path, "a", newline="") as log_file:
                writer = csv.writer(log_file)
                writer.writerow([latency, energy, mc, total_cost])
            
            return total_cost
        except Exception as e:
            print(f"评估失败: {e}")
            return float("inf")
    
    return objective

# ======================= 同构到异构的转换 =======================
def convert_homo_to_hetero(homo_trials):
    """将同构优化的结果转换为异构优化的历史记录"""
    # 先构建异构搜索空间以获取所有参数名
    hetero_space, _ = build_hetero_space()
    all_params = list(hetero_space.keys())
    
    hetero_trials = Trials()
    
    # 遍历同构优化的所有试验
    for trial_idx, trial in enumerate(homo_trials.trials):
        if trial['state'] != 2:  # 只处理成功的试验
            continue
            
        # 获取同构参数
        homo_params = {}
        for key in trial['misc']['vals']:
            if trial['misc']['vals'][key]:
                homo_params[key] = trial['misc']['vals'][key][0]
        
        # 转换参数到异构格式
        hetero_params = {}
        num_chiplets_idx = homo_params['chiplet_count_options']
        num_chiplets = chiplet_count_options[num_chiplets_idx]
        
        # 转换全局参数
        hetero_params['config_index'] = num_chiplets_idx
        hetero_params['nop_bw'] = homo_params['nop_bw']
        hetero_params['dram_bw'] = homo_params['dram_bw']
        hetero_params['micro_batch'] = homo_params['micro_batch']
        
        # 转换每个芯粒的参数 - 使用与build_hetero_space相同的命名规则
        for i in range(num_chiplets):
            suffix = f"_n{num_chiplets}_{i}"  # 与 build_hetero_space 中的格式一致
            hetero_params[f"type{suffix}"] = homo_params['chiplet_type']
            hetero_params[f"buffer{suffix}"] = homo_params['buffer']
            hetero_params[f"compute{suffix}"] = homo_params['compute']
        
        # 创建包含所有参数的完整字典
        full_hetero_params = {param: 0 for param in all_params}  # 初始化为默认值0
        full_hetero_params.update(hetero_params)  # 用实际值更新
        
        # 关键修复：创建符合 Hyperopt 格式的 vals 和 idxs
        vals_dict = {}
        idxs_dict = {}
        
        for param, value in full_hetero_params.items():
            vals_dict[param] = [value]  # 值转换为列表
            # 索引设置为当前试验 ID 的列表
            idxs_dict[param] = [trial_idx]  # 关键修改：使用当前试验 ID
            
        # 创建完整的异构试验记录
        hetero_trial = {
            'state': 2,  # 已完成
            'tid': trial_idx,
            'owner': None,
            'spec': None,
            'result': trial['result'],
            'misc': {
                'tid': trial_idx,
                'vals': vals_dict,
                'idxs': idxs_dict,
                'cmd': ('domain_attachment', 'FMinIter_Domain'),
                'workdir': None
            },
            'exp_key': None,
            'book_time': trial.get('book_time', None),
            'refresh_time': trial.get('refresh_time', None)
        }
        
        # 添加到异构试验

        hetero_trials.insert_trial_docs([hetero_trial])
        hetero_trials.refresh()

    
    return hetero_trials

# ======================= 结果输出 =======================
def save_best_config(best, chiplet_configs, directory, is_hetero=True):
    if is_hetero:
        config_index = best["config_index"]
        config = chiplet_configs[config_index]
        num_chiplets = config["num_chiplets"]
        
        config_json = {
            "num_chiplets": num_chiplets,
            "nop_bw": nop_bw_options[best["nop_bw"]],
            "dram_bw": dram_bw_options[best["dram_bw"]],
            "micro_batch": micro_batch_options[best["micro_batch"]],
            "chiplets": []
        }
        
        print("\n=== 最优配置（解码后） ===")
        print(f"芯粒数量: {num_chiplets}")
        print(f"NoP 带宽: {nop_bw_options[best['nop_bw']]} bits")
        print(f"DRAM 带宽: {dram_bw_options[best['dram_bw']]} GB/s")
        print(f"Micro-Batch Size: {micro_batch_options[best['micro_batch']]}")
        
        print(f"\n芯粒配置:")
        for i in range(num_chiplets):
            chiplet = config["chiplets"][i]
            type_idx = best.get(chiplet["type"])
            buffer_idx = best.get(chiplet["buffer"])
            compute_idx = best.get(chiplet["compute"])

            if type_idx is None or buffer_idx is None or compute_idx is None:
                continue

            type_str = chiplet_type_list[type_idx]
            buffer_val = buffer_size_list[buffer_idx]
            compute_val = compute_unit_list[compute_idx]

            print(f"  Chiplet {i}: 类型={type_str}, 缓冲区={buffer_val}KB, 计算单元={compute_val}")

            config_json["chiplets"].append({
                "type": type_str,
                "buffer_size": buffer_val,
                "compute_units": compute_val
            })
    else:
        num_chiplets = chiplet_count_options[best['chiplet_count_options']]
        
        config_json = {
            "num_chiplets": num_chiplets,
            "nop_bw": nop_bw_options[best['nop_bw']],
            "dram_bw": dram_bw_options[best['dram_bw']],
            "micro_batch": micro_batch_options[best['micro_batch']],
            "chiplets": []
        }
        
        print("\n=== 最优配置（解码后） ===")
        print(f"芯粒数量: {num_chiplets}")
        print(f"NoP 带宽: {nop_bw_options[best['nop_bw']]} bits")
        print(f"DRAM 带宽: {dram_bw_options[best['dram_bw']]} GB/s")
        print(f"Micro-Batch Size: {micro_batch_options[best['micro_batch']]}")

        print(f"\n芯粒配置:")
        print(f"芯粒类型: {chiplet_type_list[best['chiplet_type']]}")
        print(f"缓存大小: {buffer_size_list[best['buffer']]} KB")
        print(f"计算单元: {compute_unit_list[best['compute']]}")

        for _ in range(num_chiplets):
            config_json["chiplets"].append({
                "type": chiplet_type_list[best['chiplet_type']],
                "buffer_size": buffer_size_list[best['buffer']],
                "compute_units": compute_unit_list[best['compute']]
            })
    
    # 保存最优配置到文件
    with open(directory / "best_hardware.json", "w") as f:
        json.dump(config_json, f, indent=2)
    
    return config_json

# ======================= 主优化流程 =======================
def main():
    global log_id
    
    # 阶段1: 同构优化
    print("="*50)
    print("开始同构优化阶段...")
    print("="*50)
    
    log_id = 0
    homo_space = build_homo_space()
    homo_objective = create_homo_objective(homo_directory)
    homo_trials = Trials()
    
    homo_best = fmin(
        fn=homo_objective, 
        space=homo_space, 
        algo=algo, 
        max_evals=homo_rounds,
        rstate=rstate,
        trials=homo_trials
    )
    
    # 保存同构优化结果
    with open(homo_directory / 'homo_trials.pkl', "wb") as f:
        pickle.dump(homo_trials, f)
    
    # 输出同构最优配置
    save_best_config(homo_best, None, homo_directory, is_hetero=False)
    
    # 阶段2: 异构优化 (使用同构结果作为起点)
    print("\n" + "="*50)
    print("开始异构优化阶段...")
    print("="*50)
    
    log_id = 0
    hetero_space, chiplet_configs = build_hetero_space()
    hetero_objective = create_hetero_objective(chiplet_configs, hetero_directory)
    
    # 转换同构结果为异构格式
    hetero_trials = convert_homo_to_hetero(homo_trials)
    
    # 运行异构优化
    hetero_best = fmin(
        fn=hetero_objective, 
        space=hetero_space, 
        algo=algo, 
        max_evals=total_rounds,
        trials=hetero_trials,
        rstate=rstate
    )
    
    # 保存异构优化结果
    with open(hetero_directory / 'hetero_trials.pkl', "wb") as f:
        pickle.dump(hetero_trials, f)
    
    # 输出异构最优配置
    save_best_config(hetero_best, chiplet_configs, hetero_directory, is_hetero=True)
    
    print("\n" + "="*50)
    print("混合优化完成!")
    print("="*50)

if __name__ == "__main__":
    main()