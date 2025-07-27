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
import hashlib
import math
import random
from tqdm import tqdm
import sys

seed = 42

# Set random seeds for all components to ensure reproducibility
random.seed(seed)
np.random.seed(seed)

# Directory settings
now_d = Path(__file__).resolve().parent
base_directory = now_d / "exp_diff_1/Carch_Cmapping_decode_hybrid_edmc_rl_gov_2048_70B/"
if len(sys.argv) >= 2:
    base_directory = now_d / sys.argv[1]
homo_directory = base_directory / "homo_phase/"
hetero_directory = base_directory / "hetero_phase/"

# Ensure all directories exist
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

# Define granularity constant
GRANULARITY = 512  # Basic unit for all resource adjustments

# Design space options
chiplet_count_options = [1, 2, 4, 8, 16, 32, 64, 128]
if len(sys.argv) >= 4 and int(sys.argv[3])==72:
    chiplet_count_options = [1, 2, 4, 6, 12, 18, 24, 36, 72]
chiplet_type_list = ["NVDLA", "Eyeriss"]
buffer_size_list = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]  # These values are already multiples of 512
nop_bw_options = [32, 64, 128, 256] 
dram_bw_options = [16, 32, 64, 128, 256]
decode_micro_batch_options = [1, 2, 4, 8, 16, 32, 64, 128]
prefill_micro_batch_options = [1, 2, 4]
mixed_micro_batch_options = [1, 2, 3, 6, 11, 22, 33, 66]

micro_batch_options = decode_micro_batch_options
if len(sys.argv) >= 3 and sys.argv[2]=='prefill':
    micro_batch_options = prefill_micro_batch_options
elif len(sys.argv) >= 3 and sys.argv[2]=='mixed':
    micro_batch_options = mixed_micro_batch_options
elif len(sys.argv) >= 3 and sys.argv[2]=='decode':
    micro_batch_options = decode_micro_batch_options

# Optimization parameters
homo_max_rounds = 200  # Number of homogeneous optimization rounds
hetero_max_rounds = 200  # Number of heterogeneous optimization rounds
early_stop_tries = 40
init_rounds = 25

rstate = np.random.default_rng(seed)

mc_limit = 49.14685
compute_limit = 2048 // 2 * 1024  # Total compute units constraint
if len(sys.argv) >= 4:
    compute_limit = int(sys.argv[3]) // 2 * 1024  # Total compute units constraint

pe_x = 4
pe_y = 4

# For caching evaluated points
evaluated_points = {}
first_flag = True

def cost_func(latency, energy, mc):
    """Calculate cost function"""
    return latency * energy * mc / 1e9 / 1e12

def evaluate_config(config, directory, log_id):
    """
    Evaluate given configuration and return performance metrics
    :param config: Hardware configuration dictionary
    :param directory: Results save directory
    :param log_id: Log ID
    :return: (latency, energy, mc, total_cost) or (inf, inf, inf, inf) if failed
    """
    global evaluated_points, first_flag

    json_path = directory / f"hardware_params/input_{log_id}.json"
    csv_path = directory / f"search_out/output_{log_id}.csv"
    compass_out_path = directory / f"search_log/compass_{log_id}.out"
    run_cmd = now_d / "build/compass_orca"
    compass_config_path = base_directory / "compass_config_search.json"
    res_csv_path = directory / "search_results.csv"
    
    try:
        # Generate configuration hash key
        config_str = json.dumps(config, sort_keys=True)
        key = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
        
        # Check if already evaluated
        if key in evaluated_points:
            latency, energy, mc = evaluated_points[key]
            total_cost = cost_func(latency, energy, mc)
        else:
            # Save configuration to JSON file
            with open(json_path, "w") as f:
                json.dump(config, f, indent=2)
            
            # Run evaluation command
            with open(compass_out_path, "w") as outfile:
                subprocess.run([run_cmd, compass_config_path, json_path, csv_path], 
                            check=True, stdout=outfile, stderr=outfile, cwd=base_directory)
            
            # Read results
            with open(csv_path, "r") as f:
                header = f.readline()
                values = f.readline().strip().split(",")
                latency, energy, mc = map(float, values[:3])
            
            total_cost = cost_func(latency, energy, mc)
            evaluated_points[key] = (latency, energy, mc)
        
        # Initialize results file
        if first_flag:
            if os.path.exists(res_csv_path):
                os.remove(res_csv_path)
            with open(res_csv_path, "w", newline="") as log_file:
                writer = csv.writer(log_file)
                writer.writerow(["latency", "energy", "mc", "total_cost"])
            first_flag = False
        
        # Record results
        with open(res_csv_path, "a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([latency, energy, mc, total_cost])
        
        return latency, energy, mc, total_cost
    
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return float("inf"), float("inf"), float("inf"), float("inf")

def calculate_compute_units(num_chiplets):
    """Calculate compute units per chiplet based on chiplet count and ensure it's a multiple of GRANULARITY"""
    total_compute = compute_limit
    assert total_compute % (num_chiplets * pe_x * pe_y) == 0, "Total compute units must be divisible by chiplet count"
    per_chiplet = total_compute // num_chiplets
    assert per_chiplet % GRANULARITY == 0
    return per_chiplet

# ======================= Homogeneous Optimization Part =======================
def build_homo_space():
    """Build homogeneous optimization search space"""
    return {
        "chiplet_count_options": hp.choice("chiplet_count_options", list(range(len(chiplet_count_options)))),
        "chiplet_type": hp.choice("chiplet_type", list(range(len(chiplet_type_list)))),
        "nop_bw": hp.choice("nop_bw", list(range(len(nop_bw_options)))),
        "dram_bw": hp.choice("dram_bw", list(range(len(dram_bw_options)))),
        "micro_batch": hp.choice("micro_batch", list(range(len(micro_batch_options)))),
        "buffer": hp.choice("buffer", list(range(len(buffer_size_list)))),
    }

def create_homo_objective(directory):
    """Create homogeneous optimization objective function"""
    log_id = 0
    
    def objective(params):
        nonlocal log_id
        
        num_chiplets = chiplet_count_options[params["chiplet_count_options"]]
        chiplet_type = chiplet_type_list[params["chiplet_type"]]
        buffer = buffer_size_list[params["buffer"]]
        
        # Automatically calculate compute units
        compute = calculate_compute_units(num_chiplets)
        
        config = {
            "num_chiplets": num_chiplets,
            "nop_bw": nop_bw_options[params["nop_bw"]],
            "dram_bw": dram_bw_options[params["dram_bw"]],
            "micro_batch": micro_batch_options[params["micro_batch"]],
            "chiplets": []
        }
        
        for _ in range(num_chiplets):
            config["chiplets"].append({
                "type": chiplet_type,
                "buffer_size": buffer,
                "compute_units": compute
            })
        
        # Evaluate configuration
        _, _, _, total_cost = evaluate_config(config, directory, log_id)
        log_id += 1
        
        return total_cost
    
    return objective

# ======================= OR-Tools LNS Implementation =======================
class LNSOptimizer:
    """Heterogeneous fine-tuning optimizer using three operators"""
    def __init__(self, base_config, homo_best_cost, directory, max_evals=100):
        self.base_config = base_config
        self.directory = directory
        self.homo_best_cost = homo_best_cost
        self.max_evals = max_evals
        self.eval_count = 0
        self.log_id = 0
        
        # Number of chiplets
        self.num_chiplets = base_config["num_chiplets"]
        
        # Compute resource constraints
        self.total_compute = sum(c["compute_units"] for c in base_config["chiplets"])
        
        # Best configuration tracking
        self.best_config = copy.deepcopy(base_config)
        _, _, _, self.best_cost = evaluate_config(base_config, directory, self.log_id)
        self.log_id += 1
        self.eval_count += 1
        
        # Define variable keys
        self.type_vars = [f"type_{i}" for i in range(self.num_chiplets)]
        self.buffer_vars = [f"buffer_{i}" for i in range(self.num_chiplets)]
        self.compute_vars = [f"compute_{i}" for i in range(self.num_chiplets)]
        
        # Create initial solution
        self.initial_solution = {}
        for i in range(self.num_chiplets):
            chiplet = base_config["chiplets"][i]
            self.initial_solution[self.type_vars[i]] = 0 if chiplet["type"] == "NVDLA" else 1
            self.initial_solution[self.buffer_vars[i]] = chiplet["buffer_size"]
            self.initial_solution[self.compute_vars[i]] = chiplet["compute_units"]
        
        # Define three operators
        self.operator_types = ["type_change", "buffer_adjust", "compute_adjust"]
        
        # Operator statistics
        self.operator_attempts = {op: 0 for op in self.operator_types}
        self.operator_success = {op: 0 for op in self.operator_types}
        
        # Compute unit constraints
        self.granularity = GRANULARITY
    
    def optimize(self):
        """Execute custom neighborhood search optimization"""
        # Early stopping parameters
        no_improve_count = 0
        patience = early_stop_tries
        
        # Temperature parameters (for simulated annealing)
        initial_temp = self.best_cost*(0.1/3)
        cooling_rate = 0.95
        current_temp = initial_temp
        
        # Current solution and cost
        current_solution = copy.deepcopy(self.initial_solution)
        current_cost = self.best_cost
        
        # Progress bar
        pbar = tqdm(total=self.max_evals - self.eval_count, desc="Neighborhood Search Optimization")
        
        # Operator weights (initially uniform distribution)
        operator_weights = {op: 1.0 for op in self.operator_types}
        
        # Main optimization loop
        while self.eval_count < self.max_evals:
            # Adjust weights every 10 evaluations
            if self.eval_count > 0 and self.eval_count % 10 == 0:
                for op in operator_weights:
                    if self.operator_attempts[op] > 0:
                        success_rate = self.operator_success[op] / self.operator_attempts[op]
                        # Increase weight if success rate is high
                        operator_weights[op] = max(0.1, min(5.0, operator_weights[op] * (1.0 + success_rate * 0.5)))
                        pbar.write(f"Operator weight update: {op} = {operator_weights[op]:.2f} (success rate: {success_rate:.2f})")
            
            # Select operator (based on weights)
            operators = list(operator_weights.keys())
            weights = [operator_weights[op] for op in operators]
            operator = random.choices(operators, weights=weights, k=1)[0]
            
            # Record attempt
            self.operator_attempts[operator] += 1
            
            # Generate new solution
            new_solution = copy.deepcopy(current_solution)
            
            # Apply selected operator
            if operator == "type_change":
                change_desc = self.apply_type_change(new_solution)
            elif operator == "buffer_adjust":
                change_desc = self.apply_buffer_adjust(new_solution)
            elif self.num_chiplets>=2:  # compute_adjust
                change_desc = self.apply_compute_adjust(new_solution)
            
            # Convert solution dictionary to configuration
            new_config = self.solution_to_config(new_solution)
            
            try:
                # Evaluate configuration
                _, _, _, total_cost = evaluate_config(
                    new_config, self.directory, self.log_id
                )
                self.log_id += 1
                self.eval_count += 1
                pbar.update(1)
                
                # Update progress bar description
                pbar.set_postfix_str(f"{operator[:4]}: {change_desc[:20]}...")
                
                delta = total_cost - current_cost
                # Update best solution
                if total_cost < self.best_cost:
                    self.operator_success[operator] += 1
                    improvement = self.best_cost - total_cost
                    self.best_cost = total_cost
                    self.best_config = new_config
                    no_improve_count = 0
                    pbar.set_postfix_str(f"Improvement: {improvement:.4f}, Cost: {total_cost:.4f}, No improvement: 0/{patience}")
                else:
                    no_improve_count += 1
                    pbar.set_postfix_str(f"Cost: {total_cost:.4f}, No improvement: {no_improve_count}/{patience}, Accept probability: {total_cost:.2f},{current_cost:.2f},{-delta / current_temp:.2f},{math.exp(-delta / current_temp):.2f}")
                
                # Simulated annealing acceptance criterion
                if delta < 0 or random.random() < math.exp(-delta / current_temp):
                    current_solution = new_solution
                    current_cost = total_cost
                
                # Lower temperature
                current_temp *= cooling_rate
                
                # Check stop condition
                if no_improve_count >= patience:
                    print(f"No improvement for {patience} consecutive iterations, stopping optimization")
                    break
                    
            except Exception as e:
                print(f"Evaluation failed: {e}")
                continue
        
        pbar.close()
        
        # Calculate normalized cost
        normalized_cost = self.best_cost / self.homo_best_cost
        
        # Print operator statistics
        print("\nOperator Usage Statistics:")
        for op in self.operator_types:
            attempts = self.operator_attempts[op]
            success = self.operator_success[op]
            success_rate = success / attempts if attempts > 0 else 0
            print(f"  {op}: {attempts} attempts, {success} successes, success rate {success_rate:.2f}")
        
        return self.best_cost, self.best_config, normalized_cost

    def apply_type_change(self, solution):
        """Type change operator: randomly select a chiplet and change its type"""
        # Randomly select a chiplet
        idx = random.randint(0, self.num_chiplets - 1)
        
        # Get current type
        current_type = solution[self.type_vars[idx]]
        
        # Switch type (0→1 or 1→0)
        new_type = 1 - current_type
        
        # Apply change
        solution[self.type_vars[idx]] = new_type
        
        # Return operation information
        return f"Type change: Chiplet {idx} from {'NVDLA' if current_type==0 else 'Eyeriss'} to {'Eyeriss' if new_type==1 else 'NVDLA'}"

    def apply_buffer_adjust(self, solution):
        """Buffer adjustment operator: randomly select a chiplet and change its buffer size"""
        # Randomly select a chiplet
        idx = random.randint(0, self.num_chiplets - 1)
        
        # Get current buffer size
        current_buffer = solution[self.buffer_vars[idx]]
        
        # Randomly select new buffer size (from predefined list)
        new_buffer = GRANULARITY*random.randint(1, max(buffer_size_list)//GRANULARITY)  # Ensure it's a multiple of GRANULARITY
        
        # Apply change
        solution[self.buffer_vars[idx]] = new_buffer
        
        # Return operation information
        return f"Buffer adjustment: Chiplet {idx} from {current_buffer}KB to {new_buffer}KB"

    def apply_compute_adjust(self, solution):
        """Computing unit adjustment operator: randomly select 30% of chips and redistribute compute resources"""
        # Determine number of chips to adjust (at least 1)
        num_to_adjust = max(2, int(self.num_chiplets * 0.3))
        
        # Randomly select chiplet
        chiplet_indices = random.sample(range(self.num_chiplets), num_to_adjust)
        
        # Compute current total compute units
        total_compute = sum(solution[self.compute_vars[i]] for i in chiplet_indices)
        
        # Reallocate compute units
        new_allocations = self.redistribute_compute(total_compute, num_to_adjust)
        
        # Apply new allocation
        for i, compute_val in zip(chiplet_indices, new_allocations):
            solution[self.compute_vars[i]] = compute_val
        
        # Return operation information
        chiplet_str = ",".join(map(str, chiplet_indices))
        allocations_str = ",".join(map(str, new_allocations))
        return f"Computing adjustment: {num_to_adjust} chips[{chiplet_str}] New allocation: [{allocations_str}]"

    def redistribute_compute(self, total_compute, num_chiplets):
        """Reallocate compute resources for specified chips"""
        # Ensure allocation is a multiple of GRANULARITY
        total_compute = (total_compute // self.granularity) * self.granularity
        
        # Create initial allocation (uniform allocation)
        base_alloc = total_compute // num_chiplets
        # Ensure base allocation is a multiple of GRANULARITY
        base_alloc = (base_alloc // self.granularity) * self.granularity
        
        allocations = [base_alloc] * num_chiplets
        
        # Allocate remainder
        remainder = total_compute - base_alloc * num_chiplets
        for i in range(remainder // self.granularity):
            allocations[i] += self.granularity
        
        # Random perturbation (in-chiplet resource transfer)
        for _ in range(3):  # Perform 3 random transfers
            if num_chiplets < 2:
                break
                
            # Randomly select two different chips
            i, j = random.sample(range(num_chiplets), 2)
            
            # Compute maximum transfer value
            max_transfer = min(
                allocations[i] - self.granularity,  # Sender at least retain GRANULARITY
                (self.total_compute // num_chiplets) - allocations[j]  # Receiver does not exceed upper limit
            )
            
            if max_transfer >= self.granularity:
                # Randomly transfer a certain amount (GRANULARITY's multiple)
                transfer_units = random.randint(1, max_transfer // self.granularity) * self.granularity
                allocations[i] -= transfer_units
                allocations[j] += transfer_units
        
        return allocations

    def solution_to_config(self, solution):
        """Convert solution dictionary to configuration dictionary"""
        config = copy.deepcopy(self.base_config)
        config["chiplets"] = []
        
        for i in range(self.num_chiplets):
            # Get variable values
            type_val = solution[self.type_vars[i]]
            buffer_val = solution[self.buffer_vars[i]]
            compute_val = solution[self.compute_vars[i]]
            
            # Convert to configuration
            chiplet_type = "NVDLA" if type_val == 0 else "Eyeriss"
            
            # Ensure values are within reasonable range
            buffer_val = max(min(buffer_val, max(buffer_size_list)), min(buffer_size_list))
            compute_val = max(self.granularity, min(compute_val, self.total_compute))
            
            config["chiplets"].append({
                "type": chiplet_type,
                "buffer_size": int(buffer_val),
                "compute_units": int(compute_val)
            })
        
        return config

# ======================= Results Output =======================
def save_best_config(config, directory, config_type="hetero_finetune"):
    """Save best configuration to file"""
    config_json = {
        "num_chiplets": config["num_chiplets"],
        "nop_bw": config["nop_bw"],
        "dram_bw": config["dram_bw"],
        "micro_batch": config["micro_batch"],
        "chiplets": config["chiplets"]
    }
    
    # Output configuration information
    print("\n=== Best Configuration (Decoded) ===")
    print(f"Number of Chiplets: {config['num_chiplets']}")
    print(f"NoP Bandwidth: {config['nop_bw']} GB/s")
    print(f"DRAM Bandwidth: {config['dram_bw']} GB/s")
    print(f"Micro-Batch Size: {config['micro_batch']}")
    
    print(f"\nChiplet Configuration:")
    for i, chiplet in enumerate(config["chiplets"]):
        print(f"  Chiplet {i}: Type={chiplet['type']}, Buffer={chiplet['buffer_size']}KB, Compute Units={chiplet['compute_units']}")
    
    # Save to file
    with open(directory / f"best_{config_type}_hardware.json", "w") as f:
        json.dump(config_json, f, indent=2)
    
    return config_json

# ======================= Main Optimization Process =======================
def main():
    global first_flag
    # Phase 1: Homogeneous Global Optimization
    print("="*50)
    print("Starting Homogeneous Global Optimization Phase...")
    print("="*50)
    
    # Homogeneous optimization
    homo_space = build_homo_space()
    trials_path = homo_directory / 'homo_trials.pkl'
    
    # Load existing trials or create new ones
    if trials_path.exists():
        with open(trials_path, "rb") as f:
            homo_trials = pickle.load(f)
        print(f"Loaded {len(homo_trials.trials)} existing trial points")
    else:
        homo_trials = Trials()
    
    # Set early stopping parameters
    patience = early_stop_tries
    no_improve_count = 0
    best_loss = float('inf')
    
    start_time = time.time()
    # If there are existing results, set initial best value
    if any(trial['result']['status'] == STATUS_OK for trial in homo_trials.trials):
        best_loss = homo_trials.best_trial['result']['loss']
        print(f"Existing best loss: {best_loss}")
    else:
        print("No successful trial records")
    
        # Create objective function
        homo_objective = create_homo_objective(homo_directory)
        homo_algo = partial(tpe.suggest, gamma=0.5, n_startup_jobs=init_rounds)
        
        # Incremental optimization loop
        eval_count = 0
        max_evals = homo_max_rounds
        
        while eval_count < max_evals:
            eval_count += 1
            
            # Run one evaluation
            fmin(
                fn=homo_objective, 
                space=homo_space, 
                algo=homo_algo, 
                max_evals=len(homo_trials.trials) + 1,
                rstate=rstate,
                trials=homo_trials
            )
            
            # Get current best loss
            current_loss = homo_trials.best_trial['result']['loss']
            
            # Check for improvement
            if current_loss < best_loss - 1e-4:
                improvement = best_loss - current_loss
                best_loss = current_loss
                no_improve_count = 0
                print(f"Evaluation {eval_count}: Improvement found! New loss: {current_loss:.6f} (Improvement: {improvement:.6f})")
            else:
                no_improve_count += 1
                print(f"Evaluation {eval_count}: No improvement ({no_improve_count}/{patience}), Current best: {best_loss:.6f}")
            
            # Save current state
            with open(trials_path, "wb") as f:
                pickle.dump(homo_trials, f)
            
            # Check early stopping condition
            if no_improve_count >= patience:
                print(f"No improvement for {patience} consecutive times, stopping optimization")
                break
        
    # Report optimization results
    optimization_time = time.time() - start_time
    print(f"\nHomogeneous optimization completed! Total evaluations: {len(homo_trials.trials)}")
    print(f"Optimization time: {optimization_time:.2f} seconds")
    homo_best_cost = best_loss
    print(f"Best Cost: {homo_best_cost:.6f}")
    
    # Parse homogeneous optimal configuration
    from hyperopt import space_eval
    best_params = space_eval(homo_space, homo_trials.argmin)
    
    homo_config = {
        "num_chiplets": chiplet_count_options[best_params['chiplet_count_options']],
        "nop_bw": nop_bw_options[best_params['nop_bw']],
        "dram_bw": dram_bw_options[best_params['dram_bw']],
        "micro_batch": micro_batch_options[best_params['micro_batch']],
        "chiplets": []
    }
    
    # Add uniform configuration for each chiplet
    chiplet_type = chiplet_type_list[best_params['chiplet_type']]
    buffer = buffer_size_list[best_params['buffer']]
    compute = calculate_compute_units(homo_config["num_chiplets"])
    
    for _ in range(homo_config["num_chiplets"]):
        homo_config["chiplets"].append({
            "type": chiplet_type,
            "buffer_size": buffer,
            "compute_units": compute
        })
    
    # Output homogeneous optimal configuration
    save_best_config(homo_config, homo_directory, "homo")
    
    first_flag = True
    # Phase 2: Heterogeneous Fine-tuning (using OR-Tools LNS)
    print("\n" + "="*50)
    print("Starting Heterogeneous Fine-tuning Phase (using OR-Tools LNS)...")
    print("="*50)
    print(f"Fixed Global Parameters: Chiplet Count={homo_config['num_chiplets']}, NoP Bandwidth={homo_config['nop_bw']}, "
          f"DRAM Bandwidth={homo_config['dram_bw']}, Micro-Batch={homo_config['micro_batch']}")
    print(f"Homogeneous Optimal Cost: {homo_best_cost:.6f}")
    
    # Use LNS for heterogeneous fine-tuning
    lns_optimizer = LNSOptimizer(
        homo_config,
        homo_best_cost,
        hetero_directory,
        hetero_max_rounds
    )
    
    hetero_best_cost, hetero_best_config, hetero_normalized_cost = lns_optimizer.optimize()
    
    # Output heterogeneous optimal configuration
    save_best_config(hetero_best_config, hetero_directory, "hetero")
    
    # Evaluate final configuration
    print("\n" + "="*50)
    print(f"Heterogeneous Fine-tuning completed!")
    print(f"Homogeneous Optimal Cost: {homo_best_cost:.6f}")
    print(f"Heterogeneous Optimal Cost: {hetero_best_cost:.6f}")
    print(f"Normalized Cost: {hetero_normalized_cost:.6f}")
    
    improvement = homo_best_cost - hetero_best_cost
    if hetero_best_cost < homo_best_cost:
        improvement_percentage = (homo_best_cost - hetero_best_cost) / homo_best_cost * 100
        print(f"Improvement: {improvement:.6f} ({improvement_percentage:.2f}%)")
    else:
        print(f"No improvement: Heterogeneous cost is higher than Homogeneous cost {improvement:.6f}")
    
    print("="*50)

if __name__ == "__main__":
    main()