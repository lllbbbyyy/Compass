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

// Individual structure
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
    std::mutex mtx; // For protecting shared resources

    std::vector<std::vector<std::unique_ptr<CompassModelEngine>>> engines;

    unsigned int thread_num;

public:
    // Static member variables
    static int pop_size;    // Population size
    static int generations; // Number of generations

    Individual best_solution;
    double best_fitness = -1e9;
    std::vector<cycle_t> process_latency;
    std::vector<energy_t> process_energy;

    // Constructor
    GA(const std::vector<std::vector<std::shared_ptr<Network>>> &_batchedModels, const std::vector<std::shared_ptr<CoreMapper>> &_coreMappers, std::shared_ptr<NoC> _noc)
    {
        BATCH_SIZE = _batchedModels[0].size();
        LAYER_NUM = _batchedModels[0][0]->len();
        CHIPLET_NUM = _coreMappers.size();
        population.resize(pop_size);
        engines.resize(_batchedModels.size());

        auto [segmentation, layerToChip] = random_mapping(BATCH_SIZE, LAYER_NUM, CHIPLET_NUM);

        for(size_t i:tqdm(_batchedModels,"GA: create engines"))
        {
            auto model_engine = std::make_unique<CompassModelEngine>(_batchedModels[i], _coreMappers, _noc, segmentation, layerToChip);

            unsigned int n = std::thread::hardware_concurrency();
            thread_num = std::max(1u, n);
            for (size_t _=0;_<thread_num;_++)
            {
                (void)_;
                engines[i].emplace_back(std::make_unique<CompassModelEngine>(*model_engine));
            }
        }
    }

    // Parallel fitness evaluation
    void evaluate_population_parallel(std::vector<Individual> &pop);

    // Initialize population
    void initialize_population();

    void update_best_solution(std::vector<Individual> &population);

    // Crossover operation
    Individual crossover(const Individual &p1, const Individual &p2);

    // Mutation operation
    void mutate(Individual &ind);

    // Main loop
    void run();

    void random_run();

    std::tuple<cycle_t, energy_t, double, mc_t> get_best_res();

    void save_latency_detail(const std::string &filename);

    void save_energy_detail(const std::string &filename);

    void save_mc_detail(const std::string &filename);

    void save_best_solution(const std::string &filename,int micro_batch_size);

    void save_progress(const std::string &filename);

private:
    // Fitness evaluation
    void evaluate_individual(int parallel_i, Individual &ind);

    // Parallel processing of crossover and mutation
    void crossover_and_mutate_parallel();

    void crossover_and_mutate(size_t i);
    void mutate_segmentation(std::vector<int> &segmentation);
    void mutate_mapping(const std::vector<int> &segmentation, std::vector<std::vector<int>> &mapping);

    // Tournament selection operation
    Individual tournament_selection();
};

#endif // GA_H
