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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

seed = 42

"""设置所有随机组件的种子以确保可复现性"""
# Python随机模块
random.seed(seed)

# NumPy
np.random.seed(seed)

# PyTorch (CPU和GPU)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 多GPU情况

# PyTorch确定性设置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 环境变量（确保PyTorch使用确定性算法）
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# PyTorch确定性操作（可能降低性能）
torch.use_deterministic_algorithms(True)

evaluated_points = {}

# 定义粒度常量
GRANULARITY = 512  # 所有资源调整的基本单位

chiplet_count_options = [1, 2, 4, 8, 16, 32, 64, 128]
#chiplet_count_options = [1, 2, 4, 6, 12, 18, 24, 36, 72]
chiplet_type_list = ["NVDLA", "Eyeriss"]
buffer_size_list = [512, 1024, 2048, 4096, 8192, 16384]  # 这些值已经是512的倍数
nop_bw_options = [32, 64, 128, 256]
dram_bw_options = [16, 32, 64, 128, 256]
decode_micro_batch_options = [1, 2, 4, 8, 16, 32, 64, 128]
prefill_micro_batch_options = [1, 2, 4]
mixed_micro_batch_options = [1, 2, 3, 6, 11, 22, 33, 66]

micro_batch_options = mixed_micro_batch_options

# 目录设置
now_d = Path(__file__).resolve().parent
base_directory = now_d / "exp_diff_1/Carch_Cmapping_mixed_hybrid_edmc_rl_gov_512/"
homo_directory = base_directory / "homo_phase/"
hetero_directory = base_directory / "hetero_phase/"

# 优化参数
homo_max_rounds = 100  # 同构优化轮数
hetero_max_rounds = 100  # 异构优化轮数
early_stop_tries = 20
init_rounds = 25

rstate = np.random.default_rng(seed)

mc_limit = 49.14685
compute_limit = 2048 // 2 * 1024  # 总计算单元约束

pe_x = 4
pe_y = 4

def cost_func(latency, energy, mc):
    return latency * energy * mc / 1e9 / 1e12

first_flag = True
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
    }

def calculate_compute_units(num_chiplets):
    """根据芯粒数量计算每个芯粒的计算单元数，并确保是GRANULARITY的倍数"""
    total_compute = compute_limit
    assert total_compute % (num_chiplets*pe_x*pe_y) == 0, "总计算单元数必须能被芯粒数量整除"
    per_chiplet = total_compute // num_chiplets
    
    # 确保是GRANULARITY的倍数
    assert per_chiplet % GRANULARITY == 0
    return per_chiplet

def create_homo_objective(directory):
    
    log_id = 0
    
    def objective(params):
        global evaluated_points, first_flag

        nonlocal log_id
        
        num_chiplets = chiplet_count_options[params["chiplet_count_options"]]
        chiplet_type = chiplet_type_list[params["chiplet_type"]]
        buffer = buffer_size_list[params["buffer"]]
        
        # 自动计算计算单元数，确保是GRANULARITY的倍数
        compute = calculate_compute_units(num_chiplets)
        
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

            # 生成有序的json字符串作为哈希输入
            config_str = json.dumps(config_json, sort_keys=True)
            key = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
            
            if key in evaluated_points:
                latency, energy, mc = evaluated_points[key]
                total_cost = cost_func(latency, energy, mc)
            else:
                with open(compass_out_path, "w") as outfile:
                    subprocess.run([run_cmd, compass_config_path, json_path, csv_path], 
                                check=True, stdout=outfile, stderr=outfile, cwd=base_directory)
                
                with open(csv_path, "r") as f:
                    header = f.readline()
                    values = f.readline().strip().split(",")
                    latency, energy, mc = map(float, values[:3])
                total_cost = cost_func(latency, energy, mc)
            
            evaluated_points[key] = (latency, energy, mc)
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

# ======================= 强化学习策略网络 =======================
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)  # 使用新的动作空间大小
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

# ======================= 强化学习环境 =======================
class ChipletEnv:
    def __init__(self, base_config, directory):
        self.base_config = base_config
        self.directory = directory
        self.log_id = 0
        self.num_chiplets = base_config["num_chiplets"]
        self.granularity = GRANULARITY  # 使用全局粒度设置
        
        # 初始化状态
        self.state = self.config_to_state(base_config)
        self.state_dim = len(self.state)
        
        # 动作空间定义:
        # 动作0: 增加缓冲区
        # 动作1: 减少缓冲区
        # 动作2: 转移计算单元
        # 动作3: 修改芯粒类型
        # 计算完整的动作空间大小
        self.action_dim_per_type = [
            self.num_chiplets,  # 动作0: 选择芯片索引
            self.num_chiplets,  # 动作1: 选择芯片索引
            self.num_chiplets * self.num_chiplets,  # 动作2: 源芯片×目标芯片
            self.num_chiplets   # 动作3: 选择芯片索引
        ]
        self.action_dim = sum(self.action_dim_per_type)

        
    def config_to_state(self, config):
        """将配置转换为状态向量，确保缓冲区大小和计算单元是粒度的倍数"""
        state = []
        for chiplet in config["chiplets"]:
            # 类型: NVDLA=0, Eyeriss=1
            state.append(0 if chiplet["type"] == "NVDLA" else 1)
            
            # 缓冲区大小 (归一化并确保是粒度的倍数)
            buffer_val = chiplet["buffer_size"]

            state.append(buffer_val / max(buffer_size_list))
            
            # 计算单元 (归一化并确保是粒度的倍数)
            compute_val = chiplet["compute_units"]

            state.append(compute_val / compute_limit)
        return np.array(state)
    
    def state_to_config(self, state):
        """将状态向量转换回配置，确保缓冲区大小和计算单元是粒度的倍数"""
        config = copy.deepcopy(self.base_config)
        config["chiplets"] = []
        
        for i in range(self.num_chiplets):
            idx = i * 3
            # 类型
            chiplet_type = "NVDLA" if state[idx] < 0.5 else "Eyeriss"
            
            # 缓冲区大小 (反归一化并确保是粒度的倍数)
            buffer_val = state[idx+1] * max(buffer_size_list)
            # 四舍五入到最接近的粒度的倍数
            buffer_val = round(buffer_val / self.granularity) * self.granularity
            # 确保在合法范围内
            
            # 计算单元 (反归一化并确保是粒度的倍数)
            compute_val = state[idx+2] * compute_limit
            # 四舍五入到最接近的粒度的倍数
            compute_units = int(round(compute_val / self.granularity)) * self.granularity
            # 确保在合法范围内
            
            config["chiplets"].append({
                "type": chiplet_type,
                "buffer_size": int(buffer_val),
                "compute_units": int(compute_units)
            })
        
        return config

    def evaluate_config(self, config):
        """评估配置并返回性能指标"""
        global evaluated_points, first_flag
        
        json_path = self.directory / f"hardware_params/input_{self.log_id}.json"
        csv_path = self.directory / f"search_out/output_{self.log_id}.csv"
        compass_out_path = self.directory / f"search_log/compass_{self.log_id}.out"
        run_cmd = now_d / "build/compass"
        compass_config_path = base_directory / "compass_config_search.json"
        res_csv_path = self.directory / "hetero_search_results.csv"
        self.log_id += 1

        try:
            with open(json_path, "w") as f:
                json.dump(config, f, indent=2)

            config_str = json.dumps(config, sort_keys=True)
            key = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
            
            if key in evaluated_points:
                return evaluated_points[key]
            else:
                with open(compass_out_path, "w") as outfile:
                    subprocess.run([run_cmd, compass_config_path, json_path, csv_path], 
                                check=True, stdout=outfile, stderr=outfile, cwd=base_directory)
                
                with open(csv_path, "r") as f:
                    header = f.readline()
                    values = f.readline().strip().split(",")
                    latency, energy, mc = map(float, values[:3])
                total_cost = cost_func(latency, energy, mc)
                
                evaluated_points[key] = (latency, energy, mc)
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
                return latency, energy, mc
        except Exception as e:
            print(f"评估失败: {e}")
            return float("inf"), float("inf"), float("inf")
    
    def step(self, action_idx):
        """
        执行动作并返回新状态、奖励、是否结束
        动作结构: [动作类型, 参数1, 参数2]
        """
        # 解析动作
        action_type, param1, param2 = self.parse_action(action_idx)
        
        # 创建配置副本
        new_config = copy.deepcopy(self.base_config)
        new_config["chiplets"] = []
        for chiplet in self.base_config["chiplets"]:
            new_config["chiplets"].append(chiplet.copy())
        
        # 应用动作
        if action_type == 0:  # 增加缓冲区
            chiplet_idx = param1 % self.num_chiplets
            current_val = new_config["chiplets"][chiplet_idx]["buffer_size"]
            new_config["chiplets"][chiplet_idx]["buffer_size"] = current_val + self.granularity
        
        elif action_type == 1:  # 减少缓冲区
            chiplet_idx = param1 % self.num_chiplets
            current_val = new_config["chiplets"][chiplet_idx]["buffer_size"]
            if current_val > self.granularity:
                new_config["chiplets"][chiplet_idx]["buffer_size"] = current_val - self.granularity
        
        elif action_type == 2:  # 转移计算单元 (从源到目标)
            src_idx = param1 % self.num_chiplets
            dst_idx = param2 % self.num_chiplets
            
            if src_idx != dst_idx:
                # 计算可转移的最大单元数
                transfer_amount = self.granularity
                if new_config["chiplets"][src_idx]["compute_units"] > transfer_amount:
                    new_config["chiplets"][src_idx]["compute_units"] -= transfer_amount
                    new_config["chiplets"][dst_idx]["compute_units"] += transfer_amount
        
        # 移除了平衡计算单元操作（动作类型3）
        elif action_type == 3:  # 修改芯粒类型
            chiplet_idx = param1 % self.num_chiplets
            current_type = new_config["chiplets"][chiplet_idx]["type"]
            
            # 切换到另一种类型
            if current_type == "NVDLA":
                new_config["chiplets"][chiplet_idx]["type"] = "Eyeriss"
            else:
                new_config["chiplets"][chiplet_idx]["type"] = "NVDLA"
        
        # 确保所有值都是粒度的倍数
        for chiplet in new_config["chiplets"]:
            # 缓冲区大小
            buffer_val = chiplet["buffer_size"]
            chiplet["buffer_size"] = (buffer_val // self.granularity) * self.granularity
            
            # 计算单元
            compute_val = chiplet["compute_units"]
            chiplet["compute_units"] = (compute_val // self.granularity) * self.granularity
        
        # 评估新配置
        latency, energy, mc = self.evaluate_config(new_config)
        cost = cost_func(latency, energy, mc)
        reward = -cost  # 奖励是成本的负值
        
        # 更新状态和环境
        self.base_config = new_config
        self.state = self.config_to_state(new_config)
        
        # 检查是否结束 (固定步数后结束)
        done = False
        
        return self.state, reward, done
    
    def reset(self):
        """重置环境到初始状态"""
        self.base_config = copy.deepcopy(self.base_config)
        self.state = self.config_to_state(self.base_config)
        self.log_id = 0
        return self.state

    def parse_action(self, action_idx):
        """将动作索引解析为动作类型和参数"""
        # 定义动作空间边界
        boundaries = [0]
        for dim in self.action_dim_per_type:
            boundaries.append(boundaries[-1] + dim)
        
        # 确定动作类型
        for i in range(4):
            if boundaries[i] <= action_idx < boundaries[i+1]:
                action_type = i
                offset = action_idx - boundaries[i]
                break
        
        # 解析参数
        if action_type == 0 or action_type == 1 or action_type == 3:
            param1 = offset % self.num_chiplets
            param2 = None
        elif action_type == 2:
            param1 = offset // self.num_chiplets
            param2 = offset % self.num_chiplets
        
        return action_type, param1, param2

# ======================= 强化学习智能体 =======================
class ReinforceAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.01, gamma=0.99):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []
        self.action_dim = action_dim
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action_idx = m.sample()
        self.saved_log_probs.append(m.log_prob(action_idx))
        return action_idx.item()  # 返回完整动作索引
    
    def finish_episode(self):
        # 检查是否有经验数据
        if not self.rewards or not self.saved_log_probs:
            # print("警告：无经验数据，跳过策略更新")
            del self.rewards[:]
            del self.saved_log_probs[:]
            return
        
        # 确保列表长度匹配
        min_len = min(len(self.rewards), len(self.saved_log_probs))
        rewards = self.rewards[:min_len]
        log_probs = self.saved_log_probs[:min_len]
        
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        if len(returns) > 1:  # 多个样本才能标准化
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # 检查是否为空
        if not policy_loss:
            print("警告：策略损失计算为空")
            del self.rewards[:]
            del self.saved_log_probs[:]
            return
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # 清空列表
        del self.rewards[:]
        del self.saved_log_probs[:]

# ======================= 异构微调部分 (强化学习) =======================
# ======================= 异构微调部分 (强化学习) =======================
def hetero_finetune_with_rl(base_config, directory, max_evals=100):
    env = ChipletEnv(base_config, directory)
    agent = ReinforceAgent(env.state_dim, env.action_dim)
    
    # 更新策略配置
    update_cfg = {
        "points": [10, 30, 60, 90],
        "improve_threshold": 0.03,
        "min_interval": 5,
        "max_count": 8
    }
    
    # 早停参数
    patience = early_stop_tries  # 连续无改进次数阈值
    no_improve_count = 0
    best_reward = -10**9
    best_config = copy.deepcopy(base_config)
    
    update_count = 0
    last_update = 0
    last_reward = -10**9
    state = env.state
    
    # 添加进度条
    pbar = tqdm(total=max_evals, desc="异构微调")
    
    for eval_count in range(1, max_evals+1):
        pbar.update(1)
        
        # 选择动作
        action_idx = agent.select_action(state)
        
        # 执行动作
        next_state, reward, done = env.step(action_idx)
        
        # 更新状态
        state = next_state
        
        # 检查是否有改进
        if reward > best_reward + 1e-4:  # 使用小阈值防止浮点误差
            improvement = reward - best_reward
            best_reward = reward
            best_config = copy.deepcopy(env.base_config)
            no_improve_count = 0
            pbar.set_postfix_str(f"改进: {improvement:.4f}, 连续无改进: 0/{patience}")
        else:
            no_improve_count += 1
            pbar.set_postfix_str(f"连续无改进: {no_improve_count}/{patience}")
        
        # 检查是否更新策略
        update_needed = False
        improvement = (reward - last_reward) / max(1e-9, abs(last_reward))
        
        # 预设点更新
        if eval_count in update_cfg["points"]:
            update_needed = True
        
        # 显著改进更新
        elif improvement > update_cfg["improve_threshold"]:
            update_needed = True
        
        # 满足更新条件
        if update_needed:
            # 检查更新限制
            can_update = (
                (eval_count - last_update) >= update_cfg["min_interval"] and
                update_count < update_cfg["max_count"]
            )
            
            if can_update:
                agent.finish_episode()
                last_update = eval_count
                update_count += 1
                last_reward = reward
                # 打印更新信息
                print(f"\n策略更新 #{update_count} @ {eval_count}次评估")
        
        # 检查早停条件
        if no_improve_count >= patience:
            print(f"\n连续 {patience} 次无改进，停止异构微调")
            break
    
    pbar.close()
    
    # 最终更新
    agent.finish_episode()
    return (best_reward, best_config)
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
    global first_flag
    # 阶段1: 同构全局优化
    print("="*50)
    print("开始同构全局优化阶段...")
    print("="*50)
    
    # 同构优化
    homo_space = build_homo_space()
    trials_path = homo_directory / 'homo_trials.pkl'
    
    # 加载已有的trials或创建新的
    if trials_path.exists():
        with open(trials_path, "rb") as f:
            homo_trials = pickle.load(f)
        print(f"加载已有的 {len(homo_trials.trials)} 个试验点")
    else:
        homo_trials = Trials()
    
    # 设置早停参数
    patience = early_stop_tries  # 连续无改进次数阈值
    no_improve_count = 0
    best_loss = float('inf')
    
    # 如果有已有结果，设置初始最优值
    if any(trial['result']['status'] == STATUS_OK for trial in homo_trials.trials):
        best_loss = homo_trials.best_trial['result']['loss']
        print(f"已有最佳损失: {best_loss}")
    else:
        print("没有成功的试验记录")
    
    # 创建目标函数
    homo_objective = create_homo_objective(homo_directory)
    homo_algo = partial(tpe.suggest, gamma=0.5, n_startup_jobs=init_rounds)
    
    # 增量式优化循环
    eval_count = 0
    max_evals = homo_max_rounds  # 最大评估次数，防止无限循环
    start_time = time.time()
    
    while eval_count < max_evals:
        eval_count += 1
        
        # 运行一次评估
        fmin(
            fn=homo_objective, 
            space=homo_space, 
            algo=homo_algo, 
            max_evals=len(homo_trials.trials) + 1,  # 只评估一个新点
            rstate=rstate,
            trials=homo_trials
        )
        
        # 获取当前最佳损失
        current_loss = homo_trials.best_trial['result']['loss']
        
        # 检查是否有改进
        if current_loss < best_loss - 1e-4:  # 使用小阈值防止浮点误差
            improvement = best_loss - current_loss
            best_loss = current_loss
            no_improve_count = 0
            print(f"评估 {eval_count}: 发现改进! 新损失: {current_loss:.6f} (改进: {improvement:.6f})")
        else:
            no_improve_count += 1
            print(f"评估 {eval_count}: 无改进 ({no_improve_count}/{patience}), 当前最佳: {best_loss:.6f}")
        
        # 保存当前状态
        with open(trials_path, "wb") as f:
            pickle.dump(homo_trials, f)
        
        # 检查早停条件
        if no_improve_count >= patience:
            print(f"连续 {patience} 次无改进，停止优化")
            break
    
    # 报告优化结果
    optimization_time = time.time() - start_time
    print(f"\n同构优化完成! 总评估次数: {len(homo_trials.trials)}")
    print(f"优化耗时: {optimization_time:.2f}秒")
    print(f"最优损失: {best_loss:.6f}")
    homo_best_reward=best_loss
    
    # 解析同构最优配置
    from hyperopt import space_eval
    best_params = space_eval(homo_space, homo_trials.argmin)
    
    homo_config = {
        "num_chiplets": chiplet_count_options[best_params['chiplet_count_options']],
        "nop_bw": nop_bw_options[best_params['nop_bw']],
        "dram_bw": dram_bw_options[best_params['dram_bw']],
        "micro_batch": micro_batch_options[best_params['micro_batch']],
        "chiplets": []
    }
    
    # 为每个芯粒添加统一配置
    chiplet_type = chiplet_type_list[best_params['chiplet_type']]
    buffer = buffer_size_list[best_params['buffer']]
    compute = calculate_compute_units(homo_config["num_chiplets"])
    
    for _ in range(homo_config["num_chiplets"]):
        homo_config["chiplets"].append({
            "type": chiplet_type,
            "buffer_size": buffer,
            "compute_units": compute
        })
    
    # 输出同构最优配置
    save_best_config(homo_config, homo_directory, "homo")
    
    first_flag = True
    # 阶段2: 异构微调 (使用强化学习)
    print("\n" + "="*50)
    print("开始异构微调阶段 (强化学习)...")
    print("="*50)
    print(f"固定全局参数: 芯粒数量={homo_config['num_chiplets']}, NoP带宽={homo_config['nop_bw']}, "
          f"DRAM带宽={homo_config['dram_bw']}, Micro-Batch={homo_config['micro_batch']}")
    
    # 使用强化学习进行异构微调
    hetero_best_reward,hetero_best_config = hetero_finetune_with_rl(homo_config, hetero_directory, hetero_max_rounds)
    hetero_best_reward*=-1
    
    # 输出异构最优配置
    save_best_config(hetero_best_config, hetero_directory, "hetero_rl")
    
    # 评估最终配置
    
    print("\n" + "="*50)
    print(f"异构微调完成! 最终成本: {hetero_best_reward}")
    print("="*50)
    print(f"改进：{hetero_best_reward/homo_best_reward}")

if __name__ == "__main__":
    main()