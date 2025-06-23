from hyperopt import fmin, tpe, hp, Trials
import json
import subprocess
import uuid
import os
import csv
from functools import partial
import pickle
from pathlib import Path

import numpy as np

from BO_params import chiplet_count_options, chiplet_type_list, buffer_size_list, compute_unit_list, nop_bw_options, dram_bw_options, decode_micro_batch_options, prefill_micro_batch_options, mixed_micro_batch_options

micro_batch_options = mixed_micro_batch_options

now_d = Path(__file__).resolve().parent
directory = now_d / "exp_diff/Carch_Cmapping_mixed_hetero/"

rounds=200

log_id=0

seed = 42
rstate = np.random.default_rng(seed)  # numpy >= 1.17 推荐的方式

mc_limit=49.14685
def cost_func(latency, energy, mc):
    # if mc> mc_limit:
    #     return float("inf")
    # return latency*energy
    return latency*energy*(max(mc,mc_limit)/mc_limit)

algo = partial(tpe.suggest, gamma=0.5, n_startup_jobs=rounds//4)

dirs=[directory,directory/"hardware_params/",directory/"search_out/",directory/"search_log/",directory/"exec_out/"]
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# 构建搜索空间 + 配置记录（用于解码）
def build_search_space():
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

first_flag=True

# 构建目标函数（闭包形式传入配置）
def create_objective(chiplet_configs):
    def objective(params):
        config_index = params["config_index"]
        config = chiplet_configs[config_index]
        chiplets = config["chiplets"]
        num_chiplets = config["num_chiplets"]

        # 构造 JSON 数据结构
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

        # 保存 JSON 到临时文件
        global log_id
        json_path = directory/f"hardware_params/input_{log_id}.json"
        csv_path = directory/f"search_out/output_{log_id}.csv"
        compass_out_path = directory/f"search_log/compass_{log_id}.out"
        run_cmd= now_d/f"build/compass"
        compass_config_path = directory/"compass_config_search.json"
        res_csv_path = directory/"hetero_search_results.csv"
        log_id+=1

        try:
            with open(json_path, "w") as f:
                json.dump(config_json, f, indent=2)

            # 调用外部评估器.exe
            with open(compass_out_path, "w") as outfile:
                subprocess.run([run_cmd,compass_config_path, json_path, csv_path], check=True, stdout=outfile, stderr=outfile, cwd=directory)

            # 读取评估结果（latency, energy, mc）
            with open(csv_path, "r") as f:
                header = f.readline()
                values = f.readline().strip().split(",")
                latency, energy, mc = map(float, values[:3])

            total_cost= cost_func(latency, energy, mc)
            global first_flag
            # 追加保存到记录文件
            if first_flag:
                if os.path.exists(res_csv_path):
                    os.remove(res_csv_path)
                with open(res_csv_path, "w", newline="") as log_file:
                    writer = csv.writer(log_file)
                    writer.writerow(["latency", "energy", "mc","total_cost"])
                first_flag=False
            with open(res_csv_path, "a", newline="") as log_file:
                writer = csv.writer(log_file)
                writer.writerow([latency, energy, mc, total_cost])

            score = total_cost  # 或根据需要自定义目标函数
        except Exception as e:
            print(f"评估失败: {e}")
            score = float("inf")
        finally:
            # 清理临时文件
            pass
            # try:
            #     if os.path.exists(json_path):
            #         os.remove(json_path)
            #     if os.path.exists(csv_path):
            #         os.remove(csv_path)
            # except Exception as e:
            #     print(f"清理文件失败: {e}")
        return score

    return objective

# 运行优化流程
def main():
    space, chiplet_configs = build_search_space()
    objective = create_objective(chiplet_configs)

    trials= Trials()
    best = fmin(fn=objective, space=space, algo=algo, max_evals=rounds, rstate=rstate, trials=trials)

    config_index = best["config_index"]
    config = chiplet_configs[config_index]
    num_chiplets = config["num_chiplets"]

    # 构造 JSON 数据结构
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
    with open(directory/"best_hardware.json", "w") as f:
        json.dump(config_json, f, indent=2)

    with open(directory/'trials.pkl', "wb") as f:
        pickle.dump(trials, f)

if __name__ == "__main__":
    main()