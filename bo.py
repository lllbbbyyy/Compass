import os
import sys
import csv
import math
import random
import itertools
import json
import hashlib
import subprocess
from pathlib import Path

import torch
import gpytorch
from torch.distributions import Normal
from tqdm import tqdm

# ==========================================
# 0. Global environment, device settings
# ==========================================
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

def set_global_seed(seed):
    """Fix all random seeds to ensure strict experimental reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 1. Parse external command line arguments (compatible with BO_LNS.py logic)
# ==========================================
NOW_DIR = Path(__file__).resolve().parent

# --- 1.1 Parse working directory (sys.argv[1]) ---
# Default path
base_dir_str = "exp_diff_1/Carch_Cmapping_decode_hybrid_edmc_rl_gov_2048_gpt3_merged/"
if len(sys.argv) >= 2:
    base_dir_str = sys.argv[1]

BASE_DIRECTORY = NOW_DIR / base_dir_str
HETERO_DIRECTORY = BASE_DIRECTORY / "hetero_phase/"  
COMPASS_CONFIG_PATH = NOW_DIR / base_dir_str / "compass_config_search.json"

# Create all necessary data directories
for d in [
    HETERO_DIRECTORY, 
    HETERO_DIRECTORY / "hardware_params", 
    HETERO_DIRECTORY / "search_out", 
    HETERO_DIRECTORY / "search_log",
    HETERO_DIRECTORY / "exec_out", 
    ]:
    os.makedirs(d, exist_ok=True)

COMPASS_RUN_CMD = NOW_DIR / "build/compass"

LOG_FILE_PATH = HETERO_DIRECTORY / "dse_optimization_log.csv"

# --- 1.2 Parse micro-batch parameters/task type (sys.argv[2]) ---
micro_batch_options = [] # default decode
if len(sys.argv) >= 3:
    task_type = sys.argv[2]
    if task_type == 'prefill':
        micro_batch_options = [1, 2, 4]
    elif task_type == 'mixed':
        micro_batch_options = [1, 2, 3, 6, 11, 22, 33, 66]
    elif task_type == 'decode':
        micro_batch_options = [1, 2, 4, 8, 16, 32, 64, 128]
    elif task_type == 'serving_prefill':
        micro_batch_options = [1]
    elif task_type.startswith('serving'):
        micro_batch_options = [1, 2, 4, 8, 16, 32, 64, 128]
    else:
        assert False, f"Unsupported task type: {task_type}. Supported types are 'prefill', 'decode', 'mixed'."

per_chip_macs=[1024,4096,16384]
per_chip_buffer=[2048,8192,32768]

# chip_type_list=["tpu","ascend","tesla"]
# macs_list=[
#     ["[32,32]","[64,64]","[128,128]"],#tpu
#     ["[16,16,2,2]","[16,16,4,4]","[32,32,4,4]"],#ascend
#     ["[32,8,4]","[64,8,8]","[128,16,8]"]#tesla
# ]
chip_type_list=["ws","os"]
macs_list=[
    ["[4,4,8,8]","[8,8,8,8]","[8,8,16,16]"],#tpu
    ["[4,4,8,8]","[8,8,8,8]","[8,8,16,16]"],#ascend
]

shape_list=[]
if len(sys.argv) >= 4:
    scale_val = int(sys.argv[3])
    if scale_val == 64:
        shape_list=[(8, 4), (4, 2), (2,1)]
    elif scale_val == 512:
        shape_list=[(16, 16), (8, 8), (4, 4)]
    elif scale_val == 2048:
        shape_list=[(32,32), (16, 16), (8, 8)]
    else:
        assert False, f"Unsupported scale value: {scale_val}. Supported values are 32, 256, 1024."


# Simulator global cache and counter
SIMULATOR_CACHE = {}
LOG_ID_COUNTER = 0

# ==========================================
# 2. Global Unified Configuration Center (dynamically mount external parameters)
# ==========================================
SEARCH_SPACE = {
    # Dynamically generated array shape candidates: (H, W) strictly bound
    'SHAPE_LIST': shape_list, 
    
    # System-level discrete/nonlinear variables
    'SYS_PARAMS': {
        'dram_bw': [16,32,64,128,256], # DRAM bandwidth
        'nop_bw': [32,64,128,256,512],       # NoC network bandwidth
        'micro_batch': micro_batch_options
    }
}

CONFIG = {
    'GLOBAL': {
        'seed': 42,
    },
    'CHIPLET': {
        'num_types': len(chip_type_list),       # Number of chiplet types (0=small, 1=medium, 2=large)
    },
    'BO': {
        'init_samples': 10,    # Number of initial random sampling evaluations (cold start)
        'iterations': 90,     # Total iterations of the Bayesian optimization main loop
        'gp_train_steps': 20, # Number of training steps for GP model parameters after acquiring new data
        'gp_lr': 0.1,         # Learning rate for the GP model's Adam optimizer
    },
    'SA_OUTER': {             # Outer simulated annealing (searching for system-level parameters)
        'steps': 40,         # Total steps of the outer random walk
        'T_init': 1.0,        # Initial temperature
        'alpha': 0.90,        # Cooling coefficient
        'explore_decay': 0.5, # Probability decay rate for re-exploring inner layout on cache hit
    },
    'SA_INNER': {             # Inner simulated annealing (searching for architectural-level layout)
        'steps': 100,         # Total steps of the inner random walk
        'T_init': 1.0,        # Initial temperature
        'alpha': 0.95,        # Cooling coefficient
    }
}

# --- Automatically calculated dynamic dimension variables ---
NUM_SYS_VARS = len(SEARCH_SPACE['SYS_PARAMS'])
# [Modification]: The tensor prefix now has only 1 shape_norm, plus the number of system parameters
TENSOR_OFFSET = 1 + NUM_SYS_VARS 

MAX_H = int(max(shape[0] for shape in SEARCH_SPACE['SHAPE_LIST']))
MAX_W = int(max(shape[1] for shape in SEARCH_SPACE['SHAPE_LIST']))
MAX_L = MAX_H * MAX_W

# Add a helper parsing function for easy invocation anywhere
def parse_shape(shape_norm):
    """Resolve the normalized size index of [0, 1] back to the real (H, W)"""
    shape_list = SEARCH_SPACE['SHAPE_LIST']
    idx = int(round(shape_norm * (len(shape_list) - 1)))
    return shape_list[idx]

# Follow-up: class HierarchicalCompositeKernel(gpytorch.kernels.Kernel): ... 
# (Retain original core model and loop code)

# ==========================================
# 2. Real Physical Simulator Interface (Compass Simulator)
# ==========================================
def get_chiplet_spec(chip_type_idx,chip_size):
    """Map category index to physical spec dictionary required by BO_LNS.py"""
    chip_type_idx = int(chip_type_idx)
    return {
        "type": chip_type_list[chip_type_idx],
        "buffer_size": per_chip_buffer[chip_size],
        "compute_units": per_chip_macs[chip_size],
        "macs": macs_list[chip_type_idx][chip_size]
    }

def parse_tensor_to_config(x_tensor):
    x = x_tensor.detach().cpu().squeeze()
    chip_size= int(round(x[0].item() * (len(SEARCH_SPACE['SHAPE_LIST']) - 1)))
    H, W = parse_shape(x[0].item())
    num_chiplets = int(H * W)
    
    # System parameter offset starts from 1
    sys_norms = x[1:TENSOR_OFFSET]
    real_sys_vals = {}
    for i, (param_name, candidate_list) in enumerate(SEARCH_SPACE['SYS_PARAMS'].items()):
        norm_val = sys_norms[i].item()
        real_idx = int(round(norm_val * (len(candidate_list) - 1)))
        real_sys_vals[param_name] = candidate_list[real_idx]
        
    chiplets = []
    for r in range(H):
        for c in range(W):
            idx = int(r * MAX_W + c)
            chip_type = x[TENSOR_OFFSET + idx].item()
            chiplets.append(get_chiplet_spec(chip_type,chip_size))
            
    # Build JSON dictionary expected by Compass evaluation
    config = {
        "num_chiplets": num_chiplets,
        "chip_x":W,
        "chip_y":H,
        "nop_bw": real_sys_vals['nop_bw'],
        "dram_bw": real_sys_vals['dram_bw'],
        "micro_batch": real_sys_vals['micro_batch'],
        "chiplets": chiplets
    }
    return config

def CompassSimulator(x_tensor):
    global LOG_ID_COUNTER, SIMULATOR_CACHE
    
    config = parse_tensor_to_config(x_tensor)
    
    # Hash cache check
    config_str = json.dumps(config, sort_keys=True)
    key = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    if key in SIMULATOR_CACHE:
        return torch.tensor([SIMULATOR_CACHE[key]], dtype=torch.float)
        
    log_id = LOG_ID_COUNTER
    LOG_ID_COUNTER += 1
    
    json_path = HETERO_DIRECTORY / f"hardware_params/input_{log_id}.json"
    csv_path = HETERO_DIRECTORY / f"search_out/output_{log_id}.csv"
    compass_out_path = HETERO_DIRECTORY / f"search_log/compass_{log_id}.out"
    
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)
        
    try:
        # Execute underlying physical simulator command
        with open(compass_out_path, "w") as outfile:
            subprocess.run(
                [str(COMPASS_RUN_CMD), str(COMPASS_CONFIG_PATH), str(json_path), str(csv_path)], 
                check=True, stdout=outfile, stderr=outfile, cwd=BASE_DIRECTORY
            )
            
        # Read results and calculate cost
        with open(csv_path, "r") as f:
            header = f.readline()
            values = f.readline().strip().split(",")
            latency, energy, edp, mc = map(float, values[:4])
            
        total_cost = (latency * energy * mc) / 1e9 / 1e12
        
    except Exception as e:
        tqdm.write(f"\n[Warning] Simulator failed for Log ID {log_id}: {e}")
        total_cost = 1e9  # Penalty for invalid architecture
        
    SIMULATOR_CACHE[key] = total_cost
    return torch.tensor([total_cost], dtype=torch.float)

# ==========================================
# 3. Core mathematical model: dynamic composite kernel function and Gaussian process
# ==========================================
class HierarchicalCompositeKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(name="raw_lengthscale_arch", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_variance_arch", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.sys_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=TENSOR_OFFSET)
        )
        coords = torch.tensor([(i // MAX_W, i % MAX_W) for i in range(MAX_L)], dtype=torch.float)
        self.dist_matrix = torch.cdist(coords, coords, p=1)

    @property
    def lengthscale_arch(self): return torch.nn.functional.softplus(self.raw_lengthscale_arch)
    @property
    def variance_arch(self): return torch.nn.functional.softplus(self.raw_variance_arch)

    def forward(self, x1, x2, diag=False, **kwargs):
        # 1. Extract system-level parameters (now 0:TENSOR_OFFSET perfectly covers shape_idx and all sys_norms)
        sys_x1, sys_x2 = x1[:, 0:TENSOR_OFFSET], x2[:, 0:TENSOR_OFFSET]
        K_sys = self.sys_kernel(sys_x1, sys_x2, diag=diag)
        
        # 2. Extract architectural-level layout tensor
        arch_x1, arch_x2 = x1[:, TENSOR_OFFSET:], x2[:, TENSOR_OFFSET:]
        
        # Get the number of chiplet types (depends on global configuration, e.g., 3)
        num_types = CONFIG['CHIPLET']['num_types']
        
        # Get the exponential decay weight matrix of Manhattan distance
        weights = torch.exp(-self.lengthscale_arch * self.dist_matrix.to(x1.device))
        
        # ==========================================
        # Core Optimization: One-Hot + Einsum dimensionality reduction
        # ==========================================
        # Filter out invalid placeholder slots (-1)
        valid_mask1 = (arch_x1 != -1).float()
        valid_mask2 = (arch_x2 != -1).float()
        
        # Convert to One-Hot encoded matrix (shape becomes: [N, MAX_L, num_types])
        # clamp(min=0) is just to prevent passing -1 to one_hot which would cause an underlying error, it will be masked to 0 in the next step anyway
        O1 = torch.nn.functional.one_hot(arch_x1.clamp(min=0).long(), num_classes=num_types).float()
        O2 = torch.nn.functional.one_hot(arch_x2.clamp(min=0).long(), num_classes=num_types).float()
        
        # Mask One-Hot vectors of invalid slots
        O1 = O1 * valid_mask1.unsqueeze(-1)
        O2 = O2 * valid_mask2.unsqueeze(-1)
        
        if diag:
            # When diag=True, only the diagonal needs to be calculated (covariance with itself), returning a 1D vector [N]
            K_arch_diag_val = torch.einsum('ikt, kl, ilt -> i', O1, weights, O1)
            K_arch_diag = self.variance_arch * K_arch_diag_val
            return K_sys * (1.0 + K_arch_diag)
            
        # 3. Off-diagonal case: use Einsum for high-speed matrix contraction, avoiding 4D broadcasting, directly calculating [N1, N2] matrix
        K_arch_val = torch.einsum('ikt, kl, jlt -> ij', O1, weights, O2)
        
        # 4. Enforce macro-shape matching filter (aligning with your dimensionality reduction logic)
        # As long as x1 and x2 are different in shape_idx (i.e., x[:, 0]), their local architecture similarity K_arch_val is forced to zero
        shape_match = (x1[:, 0].unsqueeze(1) == x2[:, 0].unsqueeze(0)).float()
        match_mask = shape_match

        K_arch = self.variance_arch * K_arch_val * match_mask 
        
        return K_sys * (1.0 + K_arch)

class HierarchicalGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = HierarchicalCompositeKernel()
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

def expected_improvement(X_cand, model, likelihood, best_f):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        output = model(X_cand)
        mean, std = output.mean, output.variance.clamp_min(1e-9).sqrt()
        Z = (best_f - mean) / std
        normal = Normal(0, 1)
        return (best_f - mean) * normal.cdf(Z) + std * torch.exp(normal.log_prob(Z))

# ==========================================
# 4. Double-layer nested simulated annealing (inner and outer)
# ==========================================
def optimize_inner_sa(Z_fixed, model, likelihood, best_f):
    H, W = parse_shape(Z_fixed[0])
    valid_indices = [int(r * MAX_W + c) for r in range(int(H)) for c in range(int(W))]
    num_types = CONFIG['CHIPLET']['num_types']
    
    current_x = torch.full((1, TENSOR_OFFSET + MAX_L), -1.0, device=device)
    current_x[0, 0:TENSOR_OFFSET] = torch.tensor(Z_fixed, device=device)
    for idx in valid_indices:
        current_x[0, TENSOR_OFFSET + idx] = random.randint(0, num_types - 1)
        
    current_ei = expected_improvement(current_x, model, likelihood, best_f).item()
    best_x, max_ei = current_x.clone(), current_ei
    
    T, alpha, steps = CONFIG['SA_INNER']['T_init'], CONFIG['SA_INNER']['alpha'], CONFIG['SA_INNER']['steps']
    
    for _ in range(steps):
        next_x = current_x.clone()
        if random.random() < 0.5:
            idx = random.choice(valid_indices)
            next_x[0, TENSOR_OFFSET + idx] = random.randint(0, num_types - 1)
        else:
            idx1, idx2 = random.sample(valid_indices, 2)
            next_x[0, TENSOR_OFFSET + idx1], next_x[0, TENSOR_OFFSET + idx2] = next_x[0, TENSOR_OFFSET + idx2], next_x[0, TENSOR_OFFSET + idx1]
            
        next_ei = expected_improvement(next_x, model, likelihood, best_f).item()
        delta_ei = next_ei - current_ei
        if delta_ei > 0 or random.random() < math.exp(delta_ei / T):
            current_x, current_ei = next_x, next_ei
            if current_ei > max_ei:
                max_ei, best_x = current_ei, current_x.clone()
        T *= alpha
        
    return best_x, max_ei

def optimize_acqf_hierarchical(model, likelihood, best_f):
    shape_list = SEARCH_SPACE['SHAPE_LIST']
    sys_lists = list(SEARCH_SPACE['SYS_PARAMS'].values())
    
    # All lists are assembled together for processing
    all_lists = [shape_list] + sys_lists
    num_outer_vars = len(all_lists)
    
    current_indices = [random.randint(0, len(lst) - 1) for lst in all_lists]
    evaluation_cache = {}
    explore_decay = CONFIG['SA_OUTER']['explore_decay']
    
    def evaluate_outer_state(indices):
        state_key = tuple(indices)
        if state_key in evaluation_cache:
            cache_entry = evaluation_cache[state_key]
            p_explore = math.pow(explore_decay, cache_entry['hits'])
            if random.random() > p_explore:
                cache_entry['hits'] += 1
                return cache_entry['best_x'], cache_entry['max_ei']
        else:
            evaluation_cache[state_key] = {'hits': 0, 'max_ei': -float('inf'), 'best_x': None}

        Z_fixed = [idx / (len(lst) - 1) if len(lst) > 1 else 0.0 for idx, lst in zip(indices, all_lists)]
        
        cand_x, cand_ei = optimize_inner_sa(Z_fixed, model, likelihood, best_f)
        
        cache_entry = evaluation_cache[state_key]
        cache_entry['hits'] += 1
        if cand_ei > cache_entry['max_ei']:
            cache_entry['max_ei'] = cand_ei
            cache_entry['best_x'] = cand_x.clone()
        return cache_entry['best_x'], cache_entry['max_ei']

    current_best_x, current_ei = evaluate_outer_state(current_indices)
    global_max_ei, global_best_x = current_ei, current_best_x.clone()
    
    T_outer, alpha_outer, steps = CONFIG['SA_OUTER']['T_init'], CONFIG['SA_OUTER']['alpha'], CONFIG['SA_OUTER']['steps']
    
    for _ in range(steps):
        next_indices = list(current_indices)
        mutate_dim = random.randint(0, num_outer_vars - 1)
        valid_choices = list(range(len(all_lists[mutate_dim])))
        
        if len(valid_choices) > 1:
            valid_choices.remove(current_indices[mutate_dim])
            next_indices[mutate_dim] = random.choice(valid_choices)
            
        next_best_x, next_ei = evaluate_outer_state(next_indices)
        
        delta_ei = next_ei - current_ei
        if delta_ei > 0 or random.random() < math.exp(delta_ei / T_outer):
            current_indices, current_ei = next_indices, next_ei
            if current_ei > global_max_ei:
                global_max_ei, global_best_x = current_ei, next_best_x.clone()
        T_outer *= alpha_outer
        
    return global_best_x

# ==========================================
# 5. CSV Log helper parsing function
# ==========================================
def parse_tensor_for_logging(x_tensor, cost_value, iteration, num_types):
    x = x_tensor.detach().cpu().squeeze()
    H, W = parse_shape(x[0].item())
    log_data = {'Iter': iteration, 'H': H, 'W': W}
    
    sys_norms = x[1:TENSOR_OFFSET]
    for i, (param_name, candidate_list) in enumerate(SEARCH_SPACE['SYS_PARAMS'].items()):
        norm_val = sys_norms[i].item()
        real_idx = int(round(norm_val * (len(candidate_list) - 1)))
        log_data[param_name] = candidate_list[real_idx]
        
    chiplet_counts = {f'Type_{i}_Count': 0 for i in range(num_types)}
    for r in range(H):
        for c in range(W):
            idx = int(r * MAX_W + c)
            chip_type = int(x[TENSOR_OFFSET + idx].item())
            chiplet_counts[f'Type_{chip_type}_Count'] += 1
            
    log_data.update(chiplet_counts)
    log_data['Cost'] = round(cost_value, 6)
    return log_data

# ==========================================
# 6. Main program entry (including full-link closed loop)
# ==========================================
def run_hierarchical_bo():
    set_global_seed(CONFIG['GLOBAL']['seed'])
    
    print(f"--- Starting Hierarchical Chiplet Architecture DSE (Dynamic System Dimensions: {NUM_SYS_VARS}) ---")
    
    # Initialize CSV Log
    
    sys_keys = list(SEARCH_SPACE['SYS_PARAMS'].keys())
    num_types = CONFIG['CHIPLET']['num_types']
    count_keys = [f'Type_{i}_Count' for i in range(num_types)]
    fieldnames = ['Iter', 'H', 'W'] + sys_keys + count_keys + ['Cost']
    
    with open(LOG_FILE_PATH, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    train_x_list = []
    sys_lists = list(SEARCH_SPACE['SYS_PARAMS'].values())
    
    print("\n[*] Phase 1: Performing initial architecture random sampling and evaluation...")
    for _ in range(CONFIG['BO']['init_samples']):
        # [Modification]: Extract a unified index array and normalize it
        indices = [random.randint(0, len(lst)-1) for lst in [SEARCH_SPACE['SHAPE_LIST']] + sys_lists]
        Z_fixed = [idx / (len(lst) - 1) if len(lst) > 1 else 0.0 for idx, lst in zip(indices, [SEARCH_SPACE['SHAPE_LIST']] + sys_lists)]
        
        # Resolve real H and W for generating internal micro-layout
        H, W = SEARCH_SPACE['SHAPE_LIST'][indices[0]]
        
        x = torch.full((1, TENSOR_OFFSET + MAX_L), -1.0, device=device)
        x[0, 0:TENSOR_OFFSET] = torch.tensor(Z_fixed, device=device)
        
        for r in range(int(H)):
            for c in range(int(W)):
                x[0, TENSOR_OFFSET + int(r * MAX_W + c)] = random.randint(0, num_types - 1)
        train_x_list.append(x)
        
    train_x = torch.cat(train_x_list)
    train_y_list = []
    
    for i in tqdm(range(CONFIG['BO']['init_samples']), desc="Initial evaluation progress", unit="Architecture"):
        y = CompassSimulator(train_x[i:i+1]).to(device)
        train_y_list.append(y)
        
        log_data = parse_tensor_for_logging(train_x[i:i+1], y.item(), 'Init '+str(i), num_types)
        with open(LOG_FILE_PATH, mode='a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(log_data)
            
    train_y = torch.cat(train_y_list)
    print(f"Initial dataset construction complete, current minimum cost: {train_y.min().item():.6f}\n")
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    
    print("[*] Phase 2: Starting design space exploration based on Bayesian optimization...")
    pbar = tqdm(range(CONFIG['BO']['iterations']), desc="BO optimization progress", unit="Iteration")
    
    for iteration in pbar:
        best_f = train_y.min().item()
        
        model = HierarchicalGPModel(train_x, train_y, likelihood).to(device)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['BO']['gp_lr'])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        for _ in range(CONFIG['BO']['gp_train_steps']):
            optimizer.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            optimizer.step()
            
        best_next_x = optimize_acqf_hierarchical(model, likelihood, best_f)
        new_y = CompassSimulator(best_next_x).to(device)
        
        train_x = torch.cat([train_x, best_next_x])
        train_y = torch.cat([train_y, new_y])

        log_data = parse_tensor_for_logging(best_next_x, new_y.item(), iteration + CONFIG['BO']['init_samples'], num_types)
        with open(LOG_FILE_PATH, mode='a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(log_data)

        current_global_best = train_y.min().item()
        pbar.set_postfix({'Current cost': f"{new_y.item():.6f}", 'Global best': f"{current_global_best:.6f}"})

    pbar.close()

    best_idx = torch.argmin(train_y)
    best_config = train_x[best_idx].cpu().squeeze()
    best_config_json = parse_tensor_to_config(train_x[best_idx])
    
    print("\n" + "="*40)
    print("Exploration Complete - Optimal Chip Architecture Found")
    print("="*40)
    print(f"Minimum evaluated cost (latency*energy*mc): {train_y[best_idx].item():.6f}\n")
    
    H_best, W_best = parse_shape(best_config[0].item())
    print(f"[System-level macro configuration]")
    print(f"  ▸ Array size: {H_best}x{W_best}")
    
    for i, (param_name, candidate_list) in enumerate(SEARCH_SPACE['SYS_PARAMS'].items()):
        norm_val = best_config[1 + i].item()
        real_idx = int(round(norm_val * (len(candidate_list) - 1)))
        print(f"  ▸ {param_name}: {candidate_list[real_idx]}")
    
    print(f"\n[Architectural-level micro layout (corresponding to 0, 1, 2 specs in get_chiplet_spec)]")
    for r in range(H_best):
        row_str = []
        for c in range(W_best):
            idx = int(r * MAX_W + c)
            chip_type = int(best_config[TENSOR_OFFSET + idx].item())
            row_str.append(str(chip_type))
        print("  [" + "  ".join(row_str) + "]")
        
    print(f"\n[*] Optimization trajectory has been completely saved to: {os.path.abspath(LOG_FILE_PATH)}")

    with open(HETERO_DIRECTORY / f"best_hardware.json", "w") as f:
        json.dump(best_config_json, f, indent=2)
    print("\n[*] The optimal configuration JSON file has been saved...")

if __name__ == "__main__":
    run_hierarchical_bo()
