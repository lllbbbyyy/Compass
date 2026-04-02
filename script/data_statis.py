import os
import json
import csv

def process_data():
    # Get the directory where the current script is located
    base_dir = "."
    final_result = {}

    # Iterate over all items in the current directory
    for item in os.listdir(base_dir):
        # Process only folders, skip files (e.g., the script itself or generated output files)
        if not os.path.isdir(item) or item.startswith('.'):
            continue
        
        folder_path = item
        
        try:
            # ==========================================
            # 1. Process Latency
            # ==========================================
            csv_path = os.path.join(folder_path, "hetero_phase", "exec_out", "exec_res_hetero.csv")
            latencies = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    latencies.append(float(row['latency']))
            
            # Calculate the average and divide by 1e9
            if latencies:
                latency_avg = (sum(latencies) / len(latencies)) / 1e9
            else:
                latency_avg = 0.0

            # ==========================================
            # 2. Process Energy List
            # ==========================================
            energy_path = os.path.join(folder_path, "exec_energy_detail.json")
            with open(energy_path, 'r', encoding='utf-8') as f:
                energy_data = json.load(f)
            
            chiplet_energy_raw = 0.0
            dram_energy_raw = 0.0
            nop_energy_raw = 0.0

            # Iterate over all cores
            for core, entries in energy_data.items():
                for entry in entries:
                    chiplet_energy_raw += entry.get("calcEnergy", 0) + entry.get("ubufEnergy", 0)
                    dram_energy_raw += entry.get("dramEnergy", 0)
                    nop_energy_raw += entry.get("nocEnergy", 0)
            
            # Organize into a list and divide by 1e12
            energy_list = [
                chiplet_energy_raw / 1e12,
                dram_energy_raw / 1e12,
                nop_energy_raw / 1e12
            ]

            # ==========================================
            # 3. Process Monetary Cost List
            # ==========================================
            cost_path = os.path.join(folder_path, "exec_mc_detail.json")
            with open(cost_path, 'r', encoding='utf-8') as f:
                cost_data = json.load(f)
            
            # Extract cost_compute, cost_IO, cost_os_overall in order
            cost_list = [
                cost_data.get("cost_compute", 0),
                cost_data.get("cost_IO", 0),
                cost_data.get("cost_os_overall", 0)
            ]

            # ==========================================
            # 4. Calculate Product
            # ==========================================
            total_energy = sum(energy_list)
            total_cost = sum(cost_list)
            product = latency_avg * total_energy * total_cost

            # ==========================================
            # 5. Save Results
            # ==========================================
            final_result[item] = [
                latency_avg,
                energy_list,
                cost_list,
                product
            ]
            
            print(f"Successfully processed folder: {item}")

        except FileNotFoundError as e:
            print(f"Skipping {item}: Missing required file ({e.filename})")
        except KeyError as e:
            print(f"Skipping {item}: Data format mismatch, missing key {e}")
        except Exception as e:
            print(f"Skipping {item}: An unknown error occurred - {e}")

    # Output the results to a JSON file
    output_filename = "aggregated_results.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("{\n")
        
        lines = []
        for key, value in final_result.items():
            # json.dumps does not add extra indentation and newlines by default, which is ideal for compacting inner lists
            # ensure_ascii=False prevents garbled characters in non-ASCII folder names
            value_str = json.dumps(value, ensure_ascii=False)
            # Concatenate into JSON key-value pair format with an outer indentation of 4 spaces
            lines.append(f'    "{key}": {value_str}')
            
        # Join all items with commas and newlines
        f.write(",\n".join(lines))
        f.write("\n}")
    
    print(f"\nAll data processed! Results have been saved to {output_filename}")

if __name__ == "__main__":
    process_data()