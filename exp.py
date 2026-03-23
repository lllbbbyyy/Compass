
import json
import os
import subprocess
import shutil
import pandas as pd
import stat

config_template={
    "seed":42,
    
    "req_generator_input_length_path":"../../config/sharegpt_input_token_lens.json",
    "req_generator_output_length_path":"../../config/sharegpt_output_token_lens.json",
    "req_gen_mode":"fixed",
    "is_chunked_prefill": False,

    "req_number":3,
    "req_prefill_number":2,
    "req_decode_number":64,
    "batch_size":66,

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

    "dram_num":4,

    "run_mode":"GA",

    "best_mapping_save_path":"./best_mapping.json",
    "search_process_save_path":"./search_process.csv",

    "GA_population_size":120,
    "GA_generations":100,

    "exec_load_path":"./best_mapping.json",
    "detail_latency_save_path":"",
    "detail_energy_save_path":"",
    "detail_mc_save_path":""
}

dataset=['sharegpt','govreport']
workload=['prefill','decode']
scale=[72,512,2048]

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
    64:{
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

for s in scale:
    for d in dataset:
        for w in workload:
            config = config_template.copy()
            config["req_generator_input_length_path"] = f"../../config/{d}_input_token_lens.json"
            config["req_generator_output_length_path"] = f"../../config/{d}_output_token_lens.json"
            
            config.update(workload_req_info[w])
            
            config['model_info']=scale_model_info[s]
            
            # Save the configuration to a JSON file
            dir = f"./exp_compare/Carch_Cmapping_{w}_{d}_{s}TOPS/"
            print(f"Running experiment with config: {dir}")
            os.makedirs(dir, exist_ok=True)
            filename = f"compass_config_search.json"
            with open(dir+filename, 'w') as f:
                json.dump(config, f, indent=4)
            # create exec config
            config['seed']+=1
            config['req_number']*=10
            config['run_mode']='exec'
            config['detail_latency_save_path']='./exec_latency_detail.json'
            config['detail_energy_save_path']='./exec_energy_detail.json'
            config['detail_mc_save_path']='./exec_mc_detail.json'
            filename = f"compass_config_exec.json"
            with open(dir+filename, 'w') as f:
                json.dump(config, f, indent=4)

            with open(dir+'exp_out.log', "w") as outfile:
                subprocess.run(['python3','bo.py', dir, w, str(s)], 
                check=True, stdout=outfile, stderr=outfile)
            
            exec_file=dir+'exec.sh'
            search_file=dir+'search.sh'
            shutil.copyfile('./config/exec.sh', exec_file)
            shutil.copyfile('./config/search.sh', search_file)
            st = os.stat(exec_file)
            os.chmod(exec_file, st.st_mode | stat.S_IXUSR)
            st = os.stat(search_file)
            os.chmod(search_file, st.st_mode | stat.S_IXUSR)

            exec_type=['homo','hetero']
            exec_type=['hetero']

            results=[]
            for t in exec_type:
                with open(dir+f'{t}_phase/exec_out/search_out.log', "w") as outfile:
                    subprocess.run(['./search.sh',t], 
                    check=True, stdout=outfile, stderr=outfile, cwd=dir)
                with open(dir+f'{t}_phase/exec_out/exec_out.log', "w") as outfile:
                    subprocess.run(['./exec.sh',t], 
                    check=True, stdout=outfile, stderr=outfile, cwd=dir)
                file_path=dir+f'{t}_phase/exec_out/exec_res_{t}.csv'
                df = pd.read_csv(file_path, header=0, names=['latency', 'energy', 'mc'])
                avg_latency = df['latency'].mean()
                avg_energy = df['energy'].mean()
                avg_mc = df['mc'].mean()
                result = avg_latency*avg_energy*avg_mc
                results.append(result)
            results.append((results[0]-results[-1])/results[0]*100)
            exec_type.append('improvement')
            result_df = pd.DataFrame([results], columns=exec_type)
            result_df.to_csv(dir+'exec_compare.csv', index=False)
