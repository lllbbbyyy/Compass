from pathlib import Path
import subprocess
import json
import sys
import signal, sys
import signal
import sys
import os
import time
import traceback
def signal_handler(sig_num, frame):
    """Handle and log all signals, only exit on Ctrl+C (SIGINT)."""
    try:
        sig_name = signal.Signals(sig_num).name
    except ValueError:
        sig_name = f"UNKNOWN({sig_num})"

    print(f"\n[Signal Handler] ⚠️  Caught signal: {sig_name} ({sig_num}) at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Signal Handler] PID={os.getpid()}")
    print("[Signal Handler] Stack trace:")
    traceback.print_stack(frame)
    sys.stdout.flush()

    # Only exit gracefully on Ctrl+C
    if sig_num == signal.SIGINT:
        print("[Signal Handler] 💡 Caught Ctrl+C (SIGINT) — exiting cleanly.\n")
        sys.exit(0)
    else:
        print("[Signal Handler] Continuing execution (signal ignored)...\n")

def register_signals():
    """Register handlers for common catchable signals."""
    catchable = [signal.SIGINT, signal.SIGTERM]
    for sig in catchable:
        try:
            # Avoid changing behavior of SIGCHLD too aggressively
            signal.signal(sig, signal_handler)
        except (OSError, RuntimeError, ValueError):
            # Some signals can’t be caught on this system
            continue
register_signals()

mode=''

params=[
    ['hardware','os','ws','he'],
    ['dataset','sharegpt','cnndm'],
    ['parse','chunked_prefill','decode','prefill','mixed'],
]

tensor_parallelism_options=[8,16,32,64]
micro_batch_options={
    'decode': [1,2,4,8,16,32,64,128],
    'prefill': [1],
    'mixed': [1,2,4,8,16,32,64,128],
    'chunked_prefill': [1,2,4,8,16,32,64,128]
}


config_template={
    "seed": 42,
    "req_generator_input_length_path": "config/placeholder_input_token_lens.json",
    "req_generator_output_length_path": "config/placeholder_output_token_lens.json",
    "req_gen_mode": "fixed",
    "is_chunked_prefill": False,
    "req_number": 30,
    "req_prefill_number": 4,
    "req_decode_number": 0,
    "batch_size": 4,
    "micro_batch": 2,
    "model_info": {
        "type": "gpt3",
        "n_layer": 1,
        "d_model": 4096,
        "n_head": 32,
        "d_head": 128,
        "d_ffn": 16384,
        "d_model_tiling_size": 512,
        "d_ffn_tiling_size": 2048
    },
    "dram_num": 4,
    "run_mode": "GA",
    "best_mapping_save_path": "./best_mapping.json",
    "search_process_save_path": "./search_process.csv",
    "GA_population_size": 120,
    "GA_generations": 200,
    "exec_load_path": "",
    "detail_latency_save_path": "",
    "detail_energy_save_path": "",
    "detail_mc_save_path": ""
}

model_info={
    'gpt3-6.7b':{
        "type":"gpt3",
        "n_layer":1,
        "d_model":4096,
        "n_head":32,
        "d_head":128,
        "d_ffn":16384,
        "d_model_tiling_size":512,
        "d_ffn_tiling_size":2048
    },
    'gpt3-13b':{
        "type":"gpt3",
        "n_layer":1,
        "d_model":5120,
        "n_head":40,
        "d_head":128,
        "d_ffn":20480,
        "d_model_tiling_size":640,
        "d_ffn_tiling_size":2560
    },
    'llama3-8b':{
        "type":"llama3",
        "n_layer":1,
        "d_model":4096,
        "n_head":32,
        "d_head":128,
        "n_kv_head":8,
        "d_ffn":14336,
        "d_model_tiling_size":1024,
        "d_ffn_tiling_size":3584
    }
}

parse_req_info={
    'prefill':{
        'req_number':30,
        "req_prefill_number":1,
        "req_decode_number":0,
        "batch_size":1
    },
    'decode':{
        'req_number':1,
        "req_prefill_number":0,
        "req_decode_number":128,
        "batch_size":128
    },
    'mixed':{
        'req_number':2,
        "req_prefill_number":1,
        "req_decode_number":127,
        "batch_size":128
    },
    'chunked_prefill':{
        'req_number':5,
        'is_chunked_prefill':True,
        "req_prefill_number":1,
        "req_decode_number":127,
        "batch_size":128
    }
}



chunked_groups={
    'sharegpt':4,
    'govreport':5,
    'cnndm':1
}

cnt=0
def dfs_exec(i=0,param_dict={}):
    if i==len(params):
        micro_batch_sizes=micro_batch_options[param_dict['parse']]
        for tp in tensor_parallelism_options:
            for mbs in micro_batch_sizes:
                global cnt
                cnt+=1
                # if cnt<=163:
                #     continue
                exp_dir=Path('./') / f'{param_dict["parse"]}_{param_dict["dataset"]}_{param_dict["hardware"]}' / f'tp{tp}_mbs{mbs}'
                exp_dir.mkdir(parents=True,exist_ok=True)
                root_dir='../../../'

                config=config_template.copy()
                config["req_generator_input_length_path"]=root_dir+config["req_generator_input_length_path"].replace('placeholder',param_dict['dataset'])
                config["req_generator_output_length_path"]=root_dir+config["req_generator_output_length_path"].replace('placeholder',param_dict['dataset'])
                parse_req_info['chunked_prefill']['req_number']=chunked_groups[param_dict['dataset']]
                config.update(parse_req_info[param_dict['parse']])
                config['model_info']=model_info["gpt3-6.7b"].copy()
                config['model_info']['d_model_tiling_size']=config['model_info']['d_model']//tp
                config['model_info']['d_ffn_tiling_size']=config['model_info']['d_ffn']//tp
                config['micro_batch']=mbs

                # save search config file
                if mode=='search' or mode=='all':
                    config_file='search_config.json'
                    with open(exp_dir / config_file, 'w') as f:
                        json.dump(config, f, indent=4)
                    print(f'{cnt} : Executing SEARCH experiment in {exp_dir}...',flush=True)
                    with open(exp_dir / 'exp_search_out.log', "w") as outfile:
                        subprocess.run([f'{root_dir}build/compass',config_file, f'{root_dir}config/hardware_{param_dict["hardware"]}.json', 'search_res.csv'],
                        check=True, stdout=outfile, stderr=outfile,cwd=exp_dir)

                # save exec config file
                if mode=='exec' or mode=='all':
                    config['seed']+=1
                    if param_dict['parse']!='chunked_prefill':
                        config['req_number']=300
                    config['run_mode']='exec'
                    config['exec_load_path']='./best_mapping.json'
                    config['detail_latency_save_path']='./exec_latency_detail.json'
                    config['detail_energy_save_path']='./exec_energy_detail.json'
                    config['detail_mc_save_path']='./exec_mc_detail.json'
                    config_file = f"exec_config.json"
                    with open(exp_dir / config_file, 'w') as f:
                        json.dump(config, f, indent=4)
                    print(f'Executing EXEC experiment in {exp_dir}...',flush=True)
                    with open(exp_dir / 'exp_exec_out.log', "w") as outfile:
                        subprocess.run([f'{root_dir}build/compass',config_file, f'{root_dir}config/hardware_{param_dict["hardware"]}.json', 'exec_res.csv'],
                        check=True, stdout=outfile, stderr=outfile,cwd=exp_dir)

        return

    for j,param in enumerate(params[i]):
        if j==0:
            continue
        d=param_dict.copy()
        d[params[i][0]]=param
        dfs_exec(i+1,d)


if __name__ == "__main__":
    assert len(sys.argv)==2, "Usage: python exp.py <mode>"
    mode=sys.argv[1]
    dfs_exec()