
import json
import os
import subprocess
import shutil
import pandas as pd
import stat

config_template={
    "seed":42,
    
    "req_generator_input_length_path":"../../config/govreport_input_token_lens.json",
    "req_generator_output_length_path":"../../config/govreport_output_token_lens.json",
    "req_gen_mode":"fixed",

    "req_number":5,
    "req_prefill_number":0,
    "req_decode_number":128,
    "batch_size":128,

    "model_info":{
        "type":"llama3",
        "n_layer":1,
        "d_model":8192,
        "n_head":64,
        "d_head":128,
        "n_kv_head":8,
        "d_ffn":28672,
        "d_model_tiling_size":1024,
        "d_ffn_tiling_size":3584
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


config = config_template.copy()

dir = f"./exp_hybrid_reqs/orca/"
print(f"Running experiment with config: {dir}")
os.makedirs(dir, exist_ok=True)
filename = f"compass_config_search.json"
with open(dir+filename, 'w') as f:
    json.dump(config, f, indent=4)

with open(dir+'exp_out.log', "w") as outfile:
    # !!!!please use test_compass_mapping_orca.cpp to generate the exec file first!!!!
    subprocess.run(['python3','BO_LNS.py', dir, 'decode', '2048'], 
    check=True, stdout=outfile, stderr=outfile)

