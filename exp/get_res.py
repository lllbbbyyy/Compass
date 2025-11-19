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
    """Read search_res.csv file and return latency and energy values"""
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
        print(f"Error reading file {csv_path}: {e}", file=sys.stderr)
        return None, None

def process_folder_a(folder_a_path, print_all=False):
    """
    Process a single folder A, find the folder B with the minimum latency*energy product
    
    Args:
        folder_a_path: Path to folder A
        print_all: If True, print all folder B results (sorted by product ascending)
    
    Returns:
        min_folder_b: Name of folder B with minimum product
        min_product: Minimum product value
        found_any_csv: Whether any CSV file was found
    """
    min_product = float('inf')
    min_latency=float('inf')
    min_energy=float('inf')

    min_folder_b = None
    found_any_csv = False
    results = []  # Store all results for sorting
    
    # Iterate all subfolders under folder A
    try:
        for folder_b in os.listdir(folder_a_path):
            folder_b_path = os.path.join(folder_a_path, folder_b)
            
            # Ensure it is a folder
            if not os.path.isdir(folder_b_path):
                continue
            
            # Check if exec_res.csv exists
            csv_path = os.path.join(folder_b_path, 'exec_res.csv')
            if not os.path.exists(csv_path):
                continue
            
            found_any_csv = True
            
            # Read data and calculate product
            latency, energy = read_search_res(csv_path)
            if latency is not None and energy is not None:
                product = latency * energy
                results.append((folder_b, product, latency, energy))
                
                # Update minimum
                if product < min_product:
                    min_latency=latency
                    min_energy=energy
                    min_product = product
                    min_folder_b = folder_b
    
    except Exception as e:
        print(f"Error processing folder {folder_a_path}: {e}", file=sys.stderr)
        return None, None, False
    
    # If print_all is True, print all results sorted by product ascending
    if print_all and results:
        results.sort(key=lambda x: x[1])  # Sort by product
        folder_a_name = os.path.basename(folder_a_path)
        print(f"\n  All subfolder results of {folder_a_name} (sorted by product ascending):")
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

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all folders in the same directory
    folder_list = []
    for item in os.listdir(script_dir):
        item_path = os.path.join(script_dir, item)
        # Only process folders, exclude the script file itself
        if os.path.isdir(item_path):
            folder_list.append(item)
    
    # Sort by name
    folder_list.sort()
    
    if not folder_list:
        print("No folders found")
        return
    
    # Process each folder
    print("=" * 60)
    print("Analysis results:")
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
            print(f"{folder_name}: CSV files found but no valid data could be read")
        else:
            print(f"{folder_name}: No subfolder containing exec_res.csv found")
    
    print("=" * 60)
    print("Report of chunked prefill execution results for different hardware under each dataset:")
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
            
    print("Report of vLLM mixed execution results for different hardware under each dataset:")
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

    print("Report of Orca mixed execution results for different hardware under each dataset:")
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

    print("Report of cross-dataset results for different hardware")
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
    # print("Report of vLLM mixed execution results for different hardware under each dataset:")
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