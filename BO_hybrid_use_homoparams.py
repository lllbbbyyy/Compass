from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import json
import subprocess
import os
import csv
from functools import partial
import pickle
import numpy as np
import copy
import time
from pathlib import Path

from BO_params import chiplet_count_options, chiplet_type_list, buffer_size_list, compute_unit_list, nop_bw_options, dram_bw_options, decode_micro_batch_options, prefill_micro_batch_options

micro_batch_options = prefill_micro_batch_options

# 目录设置
now_d = Path(__file__).resolve().parent
base_directory = now_d / "exp_diff/Carch_Cmapping_decode_hybrid/"
homo_directory = base_directory / "homo_phase/"
hetero_directory = base_directory / "hetero_phase/"

# 优化参数
homo_rounds = 500  # 同构优化轮数
hetero_rounds = 500  # 异构优化轮数
init_rounds = 50

seed = 42
rstate = np.random.default_rng(seed)

mc_limit = 49.14685

def cost_func(latency, energy, mc):
    return latency * energy * (max(mc, mc_limit) / mc_limit)

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
    log_id = 0
    
    def objective(params):
        nonlocal log_id, first_flag
        
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

# ======================= 异构微调部分 =======================
def build_hetero_finetune_space(fixed_config):
    """
    构建异构微调的搜索空间
    fixed_config: 第一阶段得到的最优配置（包含全局参数）
    """
    num_chiplets = fixed_config["num_chiplets"]
    space = {}
    
    # 为每个芯粒创建独立的配置空间
    for i in range(num_chiplets):
        space[f"type_{i}"] = hp.choice(f"type_{i}", list(range(len(chiplet_type_list))))
        space[f"buffer_{i}"] = hp.choice(f"buffer_{i}", list(range(len(buffer_size_list))))
        space[f"compute_{i}"] = hp.choice(f"compute_{i}", list(range(len(compute_unit_list))))
    
    return space

def create_hetero_finetune_objective(fixed_config, directory):
    """
    创建异构微调的目标函数
    fixed_config: 第一阶段得到的最优配置（包含全局参数）
    """
    first_flag = True
    log_id = 0
    num_chiplets = fixed_config["num_chiplets"]
    
    def objective(params):
        nonlocal log_id, first_flag
        
        # 使用固定的全局参数
        config_json = {
            "num_chiplets": num_chiplets,
            "nop_bw": fixed_config["nop_bw"],
            "dram_bw": fixed_config["dram_bw"],
            "micro_batch": fixed_config["micro_batch"],
            "chiplets": []
        }
        
        # 为每个芯粒设置独立配置
        for i in range(num_chiplets):
            type_idx = params.get(f"type_{i}")
            buffer_idx = params.get(f"buffer_{i}")
            compute_idx = params.get(f"compute_{i}")

            if type_idx is None or buffer_idx is None or compute_idx is None:
                continue

            config_json["chiplets"].append({
                "type": chiplet_type_list[type_idx],
                "buffer_size": buffer_size_list[buffer_idx],
                "compute_units": compute_unit_list[compute_idx]
            })
        
        json_path = directory / f"hardware_params/input_{log_id}.json"
        csv_path = directory / f"search_out/output_{log_id}.csv"
        compass_out_path = directory / f"search_log/compass_{log_id}.out"
        run_cmd = now_d / "build/compass"
        compass_config_path = base_directory / "compass_config_search.json"
        res_csv_path = directory / "hetero_finetune_results.csv"
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

# ======================= 结果输出 =======================
def save_best_config(config, directory, config_type="hetero_finetune"):
    """保存最优配置到文件"""
    config_json = {
        "num_chiplets": config["num_chiplets"],
        "nop_bw": config["nop_bw"],
        "dram_bw": config["dram_bw"],
        "micro_batch": config["micro_batch"],
        "chiplets": config["chiplets"]
    }
    
    # 输出配置信息
    print("\n=== 最优配置（解码后） ===")
    print(f"芯粒数量: {config['num_chiplets']}")
    print(f"NoP 带宽: {config['nop_bw']} bits")
    print(f"DRAM 带宽: {config['dram_bw']} GB/s")
    print(f"Micro-Batch Size: {config['micro_batch']}")
    
    print(f"\n芯粒配置:")
    for i, chiplet in enumerate(config["chiplets"]):
        print(f"  Chiplet {i}: 类型={chiplet['type']}, 缓冲区={chiplet['buffer_size']}KB, 计算单元={chiplet['compute_units']}")
    
    # 保存到文件
    with open(directory / f"best_{config_type}_hardware.json", "w") as f:
        json.dump(config_json, f, indent=2)
    
    return config_json

# ======================= 主优化流程 =======================
def main():
    # 阶段1: 同构全局优化
    print("="*50)
    print("开始同构全局优化阶段...")
    print("="*50)
    
    # 同构优化
    homo_space = build_homo_space()
    homo_objective = create_homo_objective(homo_directory)
    if os.path.exists(homo_directory / 'homo_trials.pkl'):
        with open(homo_directory / 'homo_trials.pkl', "rb") as f:
            homo_trials = pickle.load(f)
    else:
        homo_trials = Trials()

    homo_algo = partial(tpe.suggest, gamma=0.5, n_startup_jobs=init_rounds)
    homo_best = fmin(
        fn=homo_objective, 
        space=homo_space, 
        algo=homo_algo, 
        max_evals=homo_rounds,
        rstate=rstate,
        trials=homo_trials
    )
    
    # 保存同构优化结果
    with open(homo_directory / 'homo_trials.pkl', "wb") as f:
        pickle.dump(homo_trials, f)
    
    # 解析同构最优配置
    homo_config = {
        "num_chiplets": chiplet_count_options[homo_best['chiplet_count_options']],
        "nop_bw": nop_bw_options[homo_best['nop_bw']],
        "dram_bw": dram_bw_options[homo_best['dram_bw']],
        "micro_batch": micro_batch_options[homo_best['micro_batch']],
        "chiplets": []
    }
    
    # 为每个芯粒添加统一配置
    chiplet_type = chiplet_type_list[homo_best['chiplet_type']]
    buffer = buffer_size_list[homo_best['buffer']]
    compute = compute_unit_list[homo_best['compute']]
    
    for _ in range(homo_config["num_chiplets"]):
        homo_config["chiplets"].append({
            "type": chiplet_type,
            "buffer_size": buffer,
            "compute_units": compute
        })
    
    # 输出同构最优配置
    save_best_config(homo_config, homo_directory, "homo")
    print(f"\n同构优化完成! 最优成本: {homo_trials.best_trial['result']['loss']}")
    
    # 阶段2: 异构微调
    print("\n" + "="*50)
    print("开始异构微调阶段...")
    print("="*50)
    print(f"固定全局参数: 芯粒数量={homo_config['num_chiplets']}, NoP带宽={homo_config['nop_bw']}, "
          f"DRAM带宽={homo_config['dram_bw']}, Micro-Batch={homo_config['micro_batch']}")
    
    # 构建异构微调空间
    hetero_space = build_hetero_finetune_space(homo_config)
    hetero_objective = create_hetero_finetune_objective(homo_config, hetero_directory)
    hetero_trials = Trials()

    initial_point = {}
    for i in range(homo_config["num_chiplets"]):
        initial_point[f"type_{i}"] = homo_best['chiplet_type']
        initial_point[f"buffer_{i}"] = homo_best['buffer']
        initial_point[f"compute_{i}"] = homo_best['compute']

    # 评估初始点（同构最优解在异构空间中的表示）
    print("评估初始点（同构最优解在异构空间中）...")
    initial_loss = hetero_objective(initial_point)
    print(f"初始点成本: {initial_loss}")
    
    # 将初始点添加到试验记录中
    initial_trial = {
        'state': 2,  # 已完成
        'tid': 0,
        'owner': None,
        'spec': None,
        'result': {'loss': initial_loss, 'status': STATUS_OK},
        'misc': {
            'tid': 0,
            'vals': {k: [v] for k, v in initial_point.items()},
            'idxs': {k: [0] for k in initial_point.keys()},
            'cmd': ('domain_attachment', 'FMinIter_Domain'),
            'workdir': None
        },
        'exp_key': None,
        'book_time': time.time(),
        'refresh_time': time.time()
    }
    hetero_trials.insert_trial_docs([initial_trial])
    hetero_trials.refresh()

    # 使用更强的探索性（gamma值较高）
    hetero_algo = partial(tpe.suggest, gamma=0.5, n_startup_jobs=init_rounds)
    
    # 运行异构微调
    hetero_best = fmin(
        fn=hetero_objective, 
        space=hetero_space, 
        algo=hetero_algo, 
        max_evals=hetero_rounds,
        trials=hetero_trials,
        rstate=rstate
    )
    
    # 保存异构优化结果
    with open(hetero_directory / 'hetero_trials.pkl', "wb") as f:
        pickle.dump(hetero_trials, f)
    
    # 解析异构最优配置
    hetero_config = {
        "num_chiplets": homo_config["num_chiplets"],
        "nop_bw": homo_config["nop_bw"],
        "dram_bw": homo_config["dram_bw"],
        "micro_batch": homo_config["micro_batch"],
        "chiplets": []
    }
    
    # 为每个芯粒添加独立配置
    for i in range(hetero_config["num_chiplets"]):
        type_idx = hetero_best.get(f"type_{i}")
        buffer_idx = hetero_best.get(f"buffer_{i}")
        compute_idx = hetero_best.get(f"compute_{i}")

        if type_idx is None or buffer_idx is None or compute_idx is None:
            continue

        hetero_config["chiplets"].append({
            "type": chiplet_type_list[type_idx],
            "buffer_size": buffer_size_list[buffer_idx],
            "compute_units": compute_unit_list[compute_idx]
        })
    
    # 输出异构最优配置
    save_best_config(hetero_config, hetero_directory, "hetero")
    print(f"\n异构微调完成! 最优成本: {hetero_trials.best_trial['result']['loss']}")
    
    # 比较优化结果
    improvement = homo_trials.best_trial['result']['loss'] - hetero_trials.best_trial['result']['loss']
    improvement_percent = (improvement / homo_trials.best_trial['result']['loss']) * 100
    
    print("\n" + "="*50)
    print(f"混合优化完成! 异构微调使成本降低了 {improvement_percent:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()