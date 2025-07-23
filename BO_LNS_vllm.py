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

# 设置所有随机组件的种子以确保可复现性
random.seed(seed)
np.random.seed(seed)

# 目录设置
now_d = Path(__file__).resolve().parent
base_directory = now_d / "exp_diff_1/Carch_Cmapping_decode_hybrid_edmc_rl_gov_2048_70B/"
if len(sys.argv) >= 2:
    base_directory = now_d / sys.argv[1]
homo_directory = base_directory / "homo_phase/"
hetero_directory = base_directory / "hetero_phase/"

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

# 定义粒度常量
GRANULARITY = 512  # 所有资源调整的基本单位

# 设计空间选项

chiplet_count_options = [1, 2, 4, 8, 16, 32, 64, 128]
if len(sys.argv) >= 4 and int(sys.argv[3])==72:
    chiplet_count_options = [1, 2, 4, 6, 12, 18, 24, 36]
chiplet_type_list = ["NVDLA", "Eyeriss"]
buffer_size_list = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]  # 这些值已经是512的倍数
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

# 优化参数
homo_max_rounds = 200  # 同构优化轮数
hetero_max_rounds = 200  # 异构优化轮数
early_stop_tries = 40
init_rounds = 25

rstate = np.random.default_rng(seed)

mc_limit = 49.14685
compute_limit = 2048 // 2 * 1024  # 总计算单元约束
if len(sys.argv) >= 4:
    compute_limit = int(sys.argv[3]) // 2 * 1024  # 总计算单元约束

pe_x = 4
pe_y = 4

# 用于缓存已评估点
evaluated_points = {}
first_flag = True

def cost_func(latency, energy, mc):
    """计算成本函数"""
    return latency * energy * mc / 1e9 / 1e12

def evaluate_config(config, directory, log_id):
    """
    评估给定配置并返回性能指标
    :param config: 硬件配置字典
    :param directory: 结果保存目录
    :param log_id: 日志ID
    :return: (latency, energy, mc, total_cost) 或 (inf, inf, inf, inf) 如果失败
    """
    global evaluated_points, first_flag
    
    json_path = directory / f"hardware_params/input_{log_id}.json"
    csv_path = directory / f"search_out/output_{log_id}.csv"
    compass_out_path = directory / f"search_log/compass_{log_id}.out"
    run_cmd = now_d / "build/compass_vllm"
    if log_id >= 1000:
        compass_config_path = base_directory / "compass_config_search_prefill.json"
    else:
        compass_config_path = base_directory / "compass_config_search_decode.json"
    res_csv_path = directory / "search_results.csv"
    
    try:
        # 生成配置哈希键
        config_str = json.dumps(config, sort_keys=True)
        key = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
        
        # 检查是否已评估过
        if key in evaluated_points:
            latency, energy, mc = evaluated_points[key]
            total_cost = cost_func(latency, energy, mc)
        else:
            # 保存配置到JSON文件
            with open(json_path, "w") as f:
                json.dump(config, f, indent=2)
            
            # 运行评估命令
            with open(compass_out_path, "w") as outfile:
                subprocess.run([run_cmd, compass_config_path, json_path, csv_path], 
                            check=True, stdout=outfile, stderr=outfile, cwd=base_directory)
            
            # 读取结果
            with open(csv_path, "r") as f:
                header = f.readline()
                values = f.readline().strip().split(",")
                latency, energy, mc = map(float, values[:3])
            
            total_cost = cost_func(latency, energy, mc)
            evaluated_points[key] = (latency, energy, mc)
        
        # 初始化结果文件
        if first_flag:
            if os.path.exists(res_csv_path):
                os.remove(res_csv_path)
            with open(res_csv_path, "w", newline="") as log_file:
                writer = csv.writer(log_file)
                writer.writerow(["latency", "energy", "mc", "total_cost"])
            first_flag = False
        
        # 记录结果
        with open(res_csv_path, "a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([latency, energy, mc, total_cost])
        
        return latency, energy, mc, total_cost
    
    except Exception as e:
        print(f"评估失败: {e}")
        return float("inf"), float("inf"), float("inf"), float("inf")

def calculate_compute_units(num_chiplets):
    """根据芯粒数量计算每个芯粒的计算单元数，并确保是GRANULARITY的倍数"""
    total_compute = compute_limit
    assert total_compute % (num_chiplets * pe_x * pe_y) == 0, "总计算单元数必须能被芯粒数量整除"
    per_chiplet = total_compute // num_chiplets
    assert per_chiplet % GRANULARITY == 0
    return per_chiplet

# ======================= 同构优化部分 =======================
def build_homo_space():
    """构建同构优化的搜索空间"""
    return {
        "chiplet_count_options": hp.choice("chiplet_count_options", list(range(len(chiplet_count_options)))),
        "chiplet_type": hp.choice("chiplet_type", list(range(len(chiplet_type_list)))),
        "nop_bw": hp.choice("nop_bw", list(range(len(nop_bw_options)))),
        "dram_bw": hp.choice("dram_bw", list(range(len(dram_bw_options)))),
        "micro_batch_decode": hp.choice("micro_batch", list(range(len(decode_micro_batch_options)))),
        "buffer": hp.choice("buffer", list(range(len(buffer_size_list)))),
    }

def create_homo_objective(directory):
    """创建同构优化的目标函数"""
    log_id = 0
    
    def objective(params):
        nonlocal log_id
        
        num_chiplets = chiplet_count_options[params["chiplet_count_options"]]
        chiplet_type = chiplet_type_list[params["chiplet_type"]]
        buffer = buffer_size_list[params["buffer"]]
        
        # 自动计算计算单元数
        compute = calculate_compute_units(num_chiplets)
        
        prefill_config = {
            "num_chiplets": num_chiplets,
            "nop_bw": nop_bw_options[params["nop_bw"]],
            "dram_bw": dram_bw_options[params["dram_bw"]],
            "micro_batch": 1,
            "chiplets": []
        }

        decode_config = {
            "num_chiplets": num_chiplets,
            "nop_bw": nop_bw_options[params["nop_bw"]],
            "dram_bw": dram_bw_options[params["dram_bw"]],
            "micro_batch": decode_micro_batch_options[params["micro_batch_decode"]],
            "chiplets": []
        }
        
        for _ in range(num_chiplets):
            prefill_config["chiplets"].append({
                "type": chiplet_type,
                "buffer_size": buffer,
                "compute_units": compute
            })
            decode_config["chiplets"].append({
                "type": chiplet_type,
                "buffer_size": buffer,
                "compute_units": compute
            })
        
        # 评估配置
        l1, e1, mc1, total_cost_prefill = evaluate_config(prefill_config, directory, 1000+log_id)
        l2, e2, mc2, total_cost_decode = evaluate_config(decode_config, directory, log_id)
        log_id += 1
        
        avgl= (l1 + l2*5) / 6
        avge = (e1 + e2*5) / 6
        total_cost=cost_func(avgl, avge, mc1)
        return total_cost
    
    return objective

# ======================= OR-Tools LNS 实现 =======================
class LNSOptimizer:
    """使用三算子优化的异构微调优化器"""
    def __init__(self, base_config, homo_best_cost, directory, max_evals=100):
        self.base_config = base_config
        self.directory = directory
        self.homo_best_cost = homo_best_cost
        self.max_evals = max_evals
        self.eval_count = 0
        self.log_id = 0
        
        # 芯粒数量
        self.num_chiplets = base_config["num_chiplets"]
        
        # 计算资源约束
        self.total_compute = sum(c["compute_units"] for c in base_config["chiplets"])
        
        # 最佳配置跟踪
        self.best_config = copy.deepcopy(base_config)
        prefill_config=copy.deepcopy(base_config)
        prefill_config['micro_batch']=1
        l1, e1, mc1, _ = evaluate_config(prefill_config, directory, 1000+self.log_id)
        l2, e2, mc2, _ = evaluate_config(base_config, directory, self.log_id)
        self.best_cost=cost_func((l1+5*l2)/6,(e1+5*e2)/6,mc1)
        self.log_id += 1
        self.eval_count += 1
        
        # 定义变量键
        self.type_vars = [f"type_{i}" for i in range(self.num_chiplets)]
        self.buffer_vars = [f"buffer_{i}" for i in range(self.num_chiplets)]
        self.compute_vars = [f"compute_{i}" for i in range(self.num_chiplets)]
        
        # 创建初始解
        self.initial_solution = {}
        for i in range(self.num_chiplets):
            chiplet = base_config["chiplets"][i]
            self.initial_solution[self.type_vars[i]] = 0 if chiplet["type"] == "NVDLA" else 1
            self.initial_solution[self.buffer_vars[i]] = chiplet["buffer_size"]
            self.initial_solution[self.compute_vars[i]] = chiplet["compute_units"]
        
        # 定义三种操作符
        self.operator_types = ["type_change", "buffer_adjust", "compute_adjust"]
        
        # 操作符统计数据
        self.operator_attempts = {op: 0 for op in self.operator_types}
        self.operator_success = {op: 0 for op in self.operator_types}
        
        # 计算单元约束
        self.granularity = GRANULARITY
    
    def optimize(self):
        """执行自定义邻域搜索优化"""
        # 早停参数
        no_improve_count = 0
        patience = early_stop_tries
        
        # 温度参数（用于模拟退火）
        initial_temp = self.best_cost*(0.1/3)
        cooling_rate = 0.95
        current_temp = initial_temp
        
        # 当前解和成本
        current_solution = copy.deepcopy(self.initial_solution)
        current_cost = self.best_cost
        
        # 进度条
        pbar = tqdm(total=self.max_evals - self.eval_count, desc="邻域搜索优化")
        
        # 操作符权重（初始均匀分布）
        operator_weights = {op: 1.0 for op in self.operator_types}
        
        # 主优化循环
        while self.eval_count < self.max_evals:
            # 每10次评估调整一次权重
            if self.eval_count > 0 and self.eval_count % 10 == 0:
                for op in operator_weights:
                    if self.operator_attempts[op] > 0:
                        success_rate = self.operator_success[op] / self.operator_attempts[op]
                        # 成功率高则增加权重
                        operator_weights[op] = max(0.1, min(5.0, operator_weights[op] * (1.0 + success_rate * 0.5)))
                        pbar.write(f"操作符权重更新: {op} = {operator_weights[op]:.2f} (成功率: {success_rate:.2f})")
            
            # 选择操作符（基于权重）
            operators = list(operator_weights.keys())
            weights = [operator_weights[op] for op in operators]
            operator = random.choices(operators, weights=weights, k=1)[0]
            
            # 记录尝试
            self.operator_attempts[operator] += 1
            
            # 生成新解
            new_solution = copy.deepcopy(current_solution)
            
            # 应用选定操作符
            if operator == "type_change":
                change_desc = self.apply_type_change(new_solution)
            elif operator == "buffer_adjust":
                change_desc = self.apply_buffer_adjust(new_solution)
            elif self.num_chiplets>=2:  # compute_adjust
                change_desc = self.apply_compute_adjust(new_solution)
            
            # 将解字典转换为配置
            new_config = self.solution_to_config(new_solution)
            
            try:
                # 评估配置
                # _, _, _, total_cost = evaluate_config(
                #     new_config, self.directory, self.log_id
                # )
                prefill_config= copy.deepcopy(new_config)
                prefill_config['micro_batch']=1
                l1, e1, mc1, _ = evaluate_config(prefill_config, self.directory, 1000+self.log_id)
                l2, e2, mc2, _ = evaluate_config(new_config, self.directory, self.log_id)
                total_cost=cost_func((l1+5*l2)/6,(e1+5*e2)/6,mc1)

                self.log_id += 1
                self.eval_count += 1
                pbar.update(1)
                
                # 更新进度条描述
                pbar.set_postfix_str(f"{operator[:4]}: {change_desc[:20]}...")
                
                delta = total_cost - current_cost
                # 更新最佳解
                if total_cost < self.best_cost:
                    self.operator_success[operator] += 1
                    improvement = self.best_cost - total_cost
                    self.best_cost = total_cost
                    self.best_config = new_config
                    no_improve_count = 0
                    pbar.set_postfix_str(f"改进: {improvement:.4f}, 成本: {total_cost:.4f}, 无改进: 0/{patience}")
                else:
                    no_improve_count += 1
                    pbar.set_postfix_str(f"成本: {total_cost:.4f}, 无改进: {no_improve_count}/{patience}, 接受概率：{total_cost:.2f},{current_cost:.2f},{-delta / current_temp:.2f},{math.exp(-delta / current_temp):.2f}")
                
                # 模拟退火接受准则
                
                if delta < 0 or random.random() < math.exp(-delta / current_temp):
                    current_solution = new_solution
                    current_cost = total_cost
                    
                
                # 降低温度
                current_temp *= cooling_rate
                
                # 检查早停条件
                if no_improve_count >= patience:
                    print(f"连续 {patience} 次无改进，停止优化")
                    break
                    
            except Exception as e:
                print(f"评估失败: {e}")
                continue
        
        pbar.close()
        
        # 计算归一化成本
        normalized_cost = self.best_cost / self.homo_best_cost
        
        # 打印操作符统计
        print("\n操作符使用统计:")
        for op in self.operator_types:
            attempts = self.operator_attempts[op]
            success = self.operator_success[op]
            success_rate = success / attempts if attempts > 0 else 0
            print(f"  {op}: 尝试{attempts}次, 成功{success}次, 成功率{success_rate:.2f}")
        
        return self.best_cost, self.best_config, normalized_cost

    def apply_type_change(self, solution):
        """类型变更算子：随机选择一个芯粒变更其类型"""
        # 随机选择一个芯粒
        idx = random.randint(0, self.num_chiplets - 1)
        
        # 获取当前类型
        current_type = solution[self.type_vars[idx]]
        
        # 切换类型 (0→1 或 1→0)
        new_type = 1 - current_type
        
        # 应用变更
        solution[self.type_vars[idx]] = new_type
        
        # 返回操作信息
        return f"类型变更: 芯粒{idx}从{'NVDLA' if current_type==0 else 'Eyeriss'}切换为{'Eyeriss' if new_type==1 else 'NVDLA'}"

    def apply_buffer_adjust(self, solution):
        """存储调整算子：随机选择一个芯粒变更其存储大小"""
        # 随机选择一个芯粒
        idx = random.randint(0, self.num_chiplets - 1)
        
        # 获取当前存储大小
        current_buffer = solution[self.buffer_vars[idx]]
        
        # 随机选择新的存储大小（从预定义列表中）
        new_buffer = GRANULARITY*random.randint(1, max(buffer_size_list)//GRANULARITY)  # 确保是GRANULARITY的倍数
        
        # 应用变更
        solution[self.buffer_vars[idx]] = new_buffer
        
        # 返回操作信息
        return f"存储调整: 芯粒{idx}从{current_buffer}KB改为{new_buffer}KB"

    def apply_compute_adjust(self, solution):
        """计算单元调整算子：随机选择30%的芯粒重新分配计算资源"""
        # 确定要调整的芯粒数量 (至少1个)
        num_to_adjust = max(2, int(self.num_chiplets * 0.3))
        
        # 随机选择芯粒
        chiplet_indices = random.sample(range(self.num_chiplets), num_to_adjust)
        
        # 计算当前总计算单元
        total_compute = sum(solution[self.compute_vars[i]] for i in chiplet_indices)
        
        # 重新分配计算单元
        new_allocations = self.redistribute_compute(total_compute, num_to_adjust)
        
        # 应用新分配
        for i, compute_val in zip(chiplet_indices, new_allocations):
            solution[self.compute_vars[i]] = compute_val
        
        # 返回操作信息
        chiplet_str = ",".join(map(str, chiplet_indices))
        allocations_str = ",".join(map(str, new_allocations))
        return f"计算调整: {num_to_adjust}个芯粒[{chiplet_str}] 新分配: [{allocations_str}]"

    def redistribute_compute(self, total_compute, num_chiplets):
        """重新分配计算资源给指定的芯粒"""
        # 确保分配是粒度的倍数
        total_compute = (total_compute // self.granularity) * self.granularity
        
        # 创建初始分配（平均分配）
        base_alloc = total_compute // num_chiplets
        # 确保基本分配是粒度的倍数
        base_alloc = (base_alloc // self.granularity) * self.granularity
        
        allocations = [base_alloc] * num_chiplets
        
        # 分配余数
        remainder = total_compute - base_alloc * num_chiplets
        for i in range(remainder // self.granularity):
            allocations[i] += self.granularity
        
        # 随机扰动（在芯粒间转移计算资源）
        for _ in range(3):  # 进行3次随机转移
            if num_chiplets < 2:
                break
                
            # 随机选择两个不同的芯粒
            i, j = random.sample(range(num_chiplets), 2)
            
            # 计算可以转移的最大值
            max_transfer = min(
                allocations[i] - self.granularity,  # 发送方至少保留GRANULARITY
                (self.total_compute // num_chiplets) - allocations[j]  # 接收方不超过上限
            )
            
            if max_transfer >= self.granularity:
                # 随机转移一定量（GRANULARITY的倍数）
                transfer_units = random.randint(1, max_transfer // self.granularity) * self.granularity
                allocations[i] -= transfer_units
                allocations[j] += transfer_units
        
        return allocations

    def solution_to_config(self, solution):
        """将解字典转换为配置字典"""
        config = copy.deepcopy(self.base_config)
        config["chiplets"] = []
        
        for i in range(self.num_chiplets):
            # 获取变量值
            type_val = solution[self.type_vars[i]]
            buffer_val = solution[self.buffer_vars[i]]
            compute_val = solution[self.compute_vars[i]]
            
            # 转换为配置
            chiplet_type = "NVDLA" if type_val == 0 else "Eyeriss"
            
            # 确保值在合理范围内
            buffer_val = max(min(buffer_val, max(buffer_size_list)), min(buffer_size_list))
            compute_val = max(self.granularity, min(compute_val, self.total_compute))
            
            config["chiplets"].append({
                "type": chiplet_type,
                "buffer_size": int(buffer_val),
                "compute_units": int(compute_val)
            })
        
        return config

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
    patience = early_stop_tries
    no_improve_count = 0
    best_loss = float('inf')
    
    start_time = time.time()
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
        max_evals = homo_max_rounds
        
        while eval_count < max_evals:
            eval_count += 1
            
            # 运行一次评估
            fmin(
                fn=homo_objective, 
                space=homo_space, 
                algo=homo_algo, 
                max_evals=len(homo_trials.trials) + 1,
                rstate=rstate,
                trials=homo_trials
            )
            
            # 获取当前最佳损失
            current_loss = homo_trials.best_trial['result']['loss']
            
            # 检查是否有改进
            if current_loss < best_loss - 1e-4:
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
    homo_best_cost = best_loss
    print(f"最优成本: {homo_best_cost:.6f}")
    
    # 解析同构最优配置
    from hyperopt import space_eval
    best_params = space_eval(homo_space, homo_trials.argmin)
    
    homo_config = {
        "num_chiplets": chiplet_count_options[best_params['chiplet_count_options']],
        "nop_bw": nop_bw_options[best_params['nop_bw']],
        "dram_bw": dram_bw_options[best_params['dram_bw']],
        "micro_batch": micro_batch_options[best_params['micro_batch_decode']],
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
    # 阶段2: 异构微调 (使用OR-Tools LNS)
    print("\n" + "="*50)
    print("开始异构微调阶段 (使用OR-Tools LNS)...")
    print("="*50)
    print(f"固定全局参数: 芯粒数量={homo_config['num_chiplets']}, NoP带宽={homo_config['nop_bw']}, "
          f"DRAM带宽={homo_config['dram_bw']}, Micro-Batch={homo_config['micro_batch']}")
    print(f"同构最优成本: {homo_best_cost:.6f}")
    
    # 使用LNS进行异构微调
    lns_optimizer = LNSOptimizer(
        homo_config,
        homo_best_cost,
        hetero_directory,
        hetero_max_rounds
    )
    
    hetero_best_cost, hetero_best_config, hetero_normalized_cost = lns_optimizer.optimize()
    
    # 输出异构最优配置
    save_best_config(hetero_best_config, hetero_directory, "hetero")
    
    # 评估最终配置
    print("\n" + "="*50)
    print(f"异构微调完成!")
    print(f"同构最优成本: {homo_best_cost:.6f}")
    print(f"异构最优成本: {hetero_best_cost:.6f}")
    print(f"归一化成本: {hetero_normalized_cost:.6f}")
    
    improvement = homo_best_cost - hetero_best_cost
    if hetero_best_cost < homo_best_cost:
        improvement_percentage = (homo_best_cost - hetero_best_cost) / homo_best_cost * 100
        print(f"改进: {improvement:.6f} ({improvement_percentage:.2f}%)")
    else:
        print(f"未改进: 异构成本比同构高 {improvement:.6f}")
    
    print("="*50)

if __name__ == "__main__":
    main()