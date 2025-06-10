#ifndef GA_H
#define GA_H

#include "network.h"
#include "layer_engine.h"
#include "model_engine.h"
#include "compass/utils_compass.h"
#include "tqdm/tqdm.h"

#include <vector>
#include <random>
#include <thread>
#include <mutex>

// 个体结构
struct Individual
{
    std::vector<int> segmentation;
    std::vector<std::vector<int>> layerToChip;
    double fitness;
    cycle_t latency = 0;
    energy_t energy = 0;

    Individual(const std::vector<int> &seg, const std::vector<std::vector<int>> &mapping)
        : segmentation(seg), layerToChip(mapping), fitness(0) {};

    Individual() : segmentation(), layerToChip(), fitness(0) {};
};

class GA
{
private:


    int current_generation = 1;

    int BATCH_SIZE;
    int LAYER_NUM;
    int CHIPLET_NUM;

    std::vector<Individual> population;
    std::vector<Individual> new_population;
    std::mutex mtx; // 用于保护共享资源

    std::vector<std::unique_ptr<CompassModelEngine>> engines;

    unsigned int thread_num;

public:
    // 静态成员变量
    static int pop_size;    // 种群大小
    static int generations; // 代数

    Individual best_solution;
    double best_fitness = -1e9;
    std::vector<cycle_t> process_latency;
    std::vector<energy_t> process_energy;

    // 构造函数
    GA(const std::vector<std::shared_ptr<Network>> &_batchedModels, const std::vector<std::shared_ptr<CoreMapper>> &_coreMappers, std::shared_ptr<NoC> _noc)
    {
        BATCH_SIZE = _batchedModels.size();
        LAYER_NUM = _batchedModels[0]->len();
        CHIPLET_NUM = _coreMappers.size();

        auto [segmentation, layerToChip] = random_mapping(BATCH_SIZE, LAYER_NUM, CHIPLET_NUM);
        auto model_engine = std::make_unique<CompassModelEngine>(_batchedModels, _coreMappers, _noc, segmentation, layerToChip);

        population.resize(pop_size);
        unsigned int n = std::thread::hardware_concurrency();
        thread_num = std::max(1u, n);
        for (size_t _ : tqdm(thread_num, "GA: create engines"))
        {
            (void)_;
            engines.emplace_back(std::make_unique<CompassModelEngine>(*model_engine));
        }
    }

    // 适应度评估并行化
    void evaluate_population_parallel(std::vector<Individual> &pop);

    // 初始化种群
    void initialize_population();

    void update_best_solution(std::vector<Individual> &population);

    // 交叉操作
    Individual crossover(const Individual &p1, const Individual &p2);

    // 变异操作
    void mutate(Individual &ind);

    // 主循环
    void run();

    void random_run();

    std::tuple<cycle_t, energy_t, mc_t> get_best_res();

    void save_latency_detail(const std::string &filename);

    void save_energy_detail(const std::string &filename);

    void save_mc_detail(const std::string &filename);

    void save_best_solution(const std::string &filename,int micro_batch_size);

    void save_progress(const std::string &filename);

private:
    // 适应度评估
    void evaluate_individual(int i, Individual &ind);

    // 交叉和变异的并行处理
    void crossover_and_mutate_parallel();

    void crossover_and_mutate(size_t i);
    // void mutate_micro_batch_size(int &micro_batch_size);
    void mutate_segmentation(std::vector<int> &segmentation);
    void mutate_mapping(const std::vector<int> &segmentation, std::vector<std::vector<int>> &mapping);

    // 锦标赛选择操作
    Individual tournament_selection();
};

#endif // GA_H
