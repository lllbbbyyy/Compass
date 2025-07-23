import random
import os
import json
from tqdm import tqdm
import subprocess
import csv
from pathlib import Path
random.seed(42)

search_config_template={
    "seed":42,
    
    "req_generator_input_length_path":"../../config/sharegpt_input_token_lens.json",
    "req_generator_output_length_path":"../../config/sharegpt_output_token_lens.json",
    "req_gen_mode":"fixed",

    "req_number":3,
    "req_prefill_number":2,
    "req_decode_number":64,
    "batch_size":66,

    "model_info":{
        "tiling_size":4096,
        "type":"llm",
        "n_layer":1,
        "d_model":4096,
        "n_head":32,
        "d_head":128
    },

    "dram_num":4,

    "run_mode":"GA",

    "best_mapping_save_path":"./best_mapping.json",
    "search_process_save_path":"./search_process.csv",

    "GA_population_size":120,
    "GA_generations":200,

    "exec_load_path":"./best_mapping.json",
    "detail_latency_save_path":"",
    "detail_energy_save_path":"",
    "detail_mc_save_path":""
}


workload_req_info={
    'prefill':{
        'req_number':30,
        "req_prefill_number":4,
        "req_decode_number":0,
        "batch_size":4
    },
    'decode':{
        'req_number':1,
        "req_prefill_number":0,
        "req_decode_number":128,
        "batch_size":128
    },
    'mixed':{
        'req_number':4,
        "req_prefill_number":2,
        "req_decode_number":64,
        "batch_size":66
    }
}

scale_model_info={
    72:{
        "type":"gpt3",
        "n_layer":1,
        "d_model":4096,
        "n_head":32,
        "d_head":128,
        "d_ffn":16384,
        "d_model_tiling_size":512,
        "d_ffn_tiling_size":2048
    },
    512:{
        "type":"gpt3",
        "n_layer":1,
        "d_model":5120,
        "n_head":40,
        "d_head":128,
        "d_ffn":20480,
        "d_model_tiling_size":640,
        "d_ffn_tiling_size":2560
    },
    2048:{
        "type":"llama3",
        "n_layer":1,
        "d_model":8192,
        "n_head":64,
        "d_head":128,
        "n_kv_head":8,
        "d_ffn":28672,
        "d_model_tiling_size":1024,
        "d_ffn_tiling_size":3584
    }
}


chiplet_count_options = [1, 2, 4, 8, 16, 32, 64, 128]

chiplet_count_options_72TOPS = [1, 2, 4, 6, 12, 18, 24, 36]
chiplet_type_list = ["NVDLA", "Eyeriss"]
buffer_size_list = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]  # 这些值已经是512的倍数
nop_bw_options = [32, 64, 128, 256] 
dram_bw_options = [16, 32, 64, 128, 256]
decode_micro_batch_options = [1, 2, 4, 8, 16, 32, 64, 128]
prefill_micro_batch_options = [1, 2, 4]
mixed_micro_batch_options = [1, 2, 3, 6, 11, 22, 33, 66]


number=200

assert len(chiplet_count_options) == len(chiplet_count_options_72TOPS), "chiplet_count_options and chiplet_count_options_72TOPS must have the same length"


dataset=['sharegpt','govreport']
workload=['decode','prefill','mixed']
scale=[72,512,2048]

def cost_func(latency, energy, mc):
    """计算成本函数"""
    return latency * energy * mc / 1e9 / 1e12

now_dir=Path(__file__).resolve().parent
dir=now_dir / "DSE_hardware/"

for i in tqdm(range(number)):
    chiplet_number = random.choice(list(range(len(chiplet_count_options))))
    chiplet_type = random.choice(chiplet_type_list)
    buffer_size = random.choice(buffer_size_list)
    nop_bw = random.choice(nop_bw_options)
    dram_bw = random.choice(dram_bw_options)
    for d in dataset:
        for w in workload:
            for s in scale:
                if s==72:
                    chiplet_count=chiplet_count_options_72TOPS
                else:
                    chiplet_count=chiplet_count_options

                if w == 'decode':
                    micro_batch_size = random.choice(decode_micro_batch_options)
                elif w == 'prefill':
                    micro_batch_size = random.choice(prefill_micro_batch_options)
                else:  # mixed workload
                    micro_batch_size = random.choice(mixed_micro_batch_options)

                subdir = dir / f"{w}_{d}_{s}TOPS/hardware_params/"
                os.makedirs(subdir, exist_ok=True)
                
                config = {
                    "num_chiplets": chiplet_count[chiplet_number],
                    "nop_bw": nop_bw,
                    "dram_bw": dram_bw,
                    "micro_batch": micro_batch_size,
                    "chiplets":[]
                }
                chiplet_config = {
                        "type": chiplet_type,
                        "buffer_size": buffer_size,
                        "compute_units": s//2*1024// chiplet_count[chiplet_number]
                    }
                for j in range(chiplet_count[chiplet_number]):
                    config["chiplets"].append(chiplet_config)
                
                config_file_path=subdir / f"input_{i}.json"
                
                with open(config_file_path, 'w') as f:
                    json.dump(config, f, indent=4)
                
                ###
                config = search_config_template.copy()
                config["req_generator_input_length_path"] = f"../../config/{d}_input_token_lens.json"
                config["req_generator_output_length_path"] = f"../../config/{d}_output_token_lens.json"
                
                config.update(workload_req_info[w])
                
                config['model_info']=scale_model_info[s]
                
                # Save the configuration to a JSON file
                subdir = dir / f"{w}_{d}_{s}TOPS/"
                os.makedirs(subdir / 'search_out/', exist_ok=True)
                os.makedirs(subdir / 'search_log/', exist_ok=True)

                filename = f"compass_config_search.json"
                with open(subdir / filename, 'w') as f:
                    json.dump(config, f, indent=4)
                
                json_path = subdir / f"hardware_params/input_{i}.json"
                csv_path = subdir / f"search_out/output_{i}.csv"
                compass_out_path = subdir / f"search_log/compass_{i}.out"
                run_cmd = now_dir / "build/compass"
                compass_config_path = subdir / "compass_config_search.json"
                res_csv_path = subdir / "search_results.csv"

                with open(compass_out_path, "w") as outfile:
                    subprocess.run([run_cmd, compass_config_path, json_path, csv_path], 
                                check=True, stdout=outfile, stderr=outfile, cwd=subdir)
            
                # 读取结果
                with open(csv_path, "r") as f:
                    header = f.readline()
                    values = f.readline().strip().split(",")
                    latency, energy, mc = map(float, values[:3])
                    total_cost = cost_func(latency, energy, mc)
                
                if not os.path.exists(res_csv_path):
                    with open(res_csv_path, "w", newline="") as log_file:
                        writer = csv.writer(log_file)
                        writer.writerow(["latency", "energy", "mc", "total_cost"])
                with open(res_csv_path, "a", newline="") as log_file:
                    writer = csv.writer(log_file)
                    writer.writerow([latency, energy, mc, total_cost])