#include "compass/GA.h"
#include "compass/utils_compass.h"
#include "json.hpp"
#include "tqdm/tqdm.h"
#include "rapidcsv.h"
#include "debug.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <future>

static std::pair<std::vector<int>,
                 std::vector<std::vector<int>>>
pipeline_mapping(size_t BATCH_SIZE,
                 size_t LAYER_NUM,
                 size_t CHIPLET_NUM)
{

    std::vector<int> segmentation;
    std::vector<std::vector<cidx_t>> layerToChip;

    for (size_t j = 0; j < BATCH_SIZE; ++j)
    {
        layerToChip.emplace_back();
    }
    for (size_t i = 0; i < LAYER_NUM; i++)
    {
        if (i != LAYER_NUM - 1)
        {
            if ((i + 1) % CHIPLET_NUM == 0)
                segmentation.push_back(1);
            else
                segmentation.push_back(0);
        }
        for (size_t j = 0; j < BATCH_SIZE; ++j)
        {
            layerToChip[j].emplace_back(i % CHIPLET_NUM);
        }
    }
    return {segmentation, layerToChip};
}

static std::pair<std::vector<int>,
                 std::vector<std::vector<int>>>
init_mapping(size_t BATCH_SIZE,
                 size_t LAYER_NUM,
                 size_t CHIPLET_NUM)
{

    std::vector<int> segmentation(LAYER_NUM - 1, 1);
    std::vector<std::vector<cidx_t>> layerToChip;

    for (size_t j = 0; j < BATCH_SIZE; ++j)
    {
        layerToChip.emplace_back();
    }

    for (size_t i = 0; i < BATCH_SIZE; i++)
    {
        for (size_t j = 0; j < LAYER_NUM; ++j)
        {
            layerToChip[i].emplace_back(j % CHIPLET_NUM);
        }
    }
    return {segmentation, layerToChip};
}

int GA::pop_size = 250;   
int GA::generations = 300; 

void GA::initialize_population()
{
    auto [pipeline_seg, pipeline_laytochip] = pipeline_mapping(BATCH_SIZE, LAYER_NUM, CHIPLET_NUM);
    auto [init_seg, init_laytochip] = init_mapping(BATCH_SIZE, LAYER_NUM, CHIPLET_NUM);
    population[0] = Individual(pipeline_seg, pipeline_laytochip);
    population[1] = Individual(init_seg, init_laytochip);
    for (size_t i = 0+2; i < population.size(); i++)
    {
        auto [seg, laytochip] = random_mapping(BATCH_SIZE, LAYER_NUM, CHIPLET_NUM);
        population[i] = Individual(seg, laytochip);
    }
    DEBUG("Population initialized. Evaluating initial population...");
    evaluate_population_parallel(population); 
}

void GA::evaluate_population_parallel(std::vector<Individual> &pop)
{
    // std::vector<std::thread> threads;
    // for (auto& ind : pop) {
    //     threads.push_back(std::thread(&GA::evaluate_individual, this, std::ref(ind)));
    // }
    // for (auto& t : threads) {
    //     t.join();
    // }
    size_t total = pop.size();
    size_t index = 0;

    while (index < total)
    {
        std::vector<std::thread> threads;

        size_t batch_size = std::min(thread_num, static_cast<unsigned int>(total - index));
        for (size_t i = 0; i < batch_size; ++i)
        {
            threads.push_back(std::thread(&GA::evaluate_individual, this, i, std::ref(pop[index])));
            // evaluate_individual(i,pop[index]);
            index++;
        }

        for (auto &t : threads)
        {
            t.join();
        }
    }
    // for (auto& ind : pop) {
    //     GA::evaluate_individual(ind);
    // }
}

void GA::evaluate_individual(int parallel_i, Individual &ind)
{
    cycle_t all_model_latency = 0;
    energy_t all_model_energy = 0;
    double all_cost=0;
    for(size_t i=0;i<engines.size();++i)
    {
        engines[i][parallel_i]->setSegmentation(ind.segmentation, ind.layerToChip);
        auto [latency, energy] = engines[i][parallel_i]->calcLatencyAndEnergy();
        all_model_latency += latency;
        all_model_energy += energy;
        all_cost += cost_func(latency, energy, 1);
    }
    
    ind.latency = all_model_latency / engines.size();
    ind.energy = all_model_energy / engines.size();
    all_cost /= engines.size();
    ind.fitness = 1 / all_cost;
}

Individual GA::crossover(const Individual &p1, const Individual &p2)
{
    Individual child = p1;

    for (size_t i = 0; i < child.segmentation.size(); ++i)
        child.segmentation[i] = (ThreadSafeRandom::rand_int(0, 1) ? p1.segmentation[i] : p2.segmentation[i]);

    int start = 0;
    for (size_t s = 0; s <= child.segmentation.size(); ++s)
    {
        int end = (s == child.segmentation.size()) ? LAYER_NUM : s + 1;
        bool from_p1 = ThreadSafeRandom::rand_int(0, 1);
        for (int b = 0; b < BATCH_SIZE; ++b)
            for (int l = start; l < end; ++l)
                child.layerToChip[b][l] = from_p1 ? p1.layerToChip[b][l] : p2.layerToChip[b][l];
        if (s < p1.segmentation.size() && p1.segmentation[s] == 1)
            start = s + 1;
    }
    return child;
}

void GA::mutate(Individual &ind)
{
    mutate_segmentation(ind.segmentation);
    mutate_mapping(ind.segmentation, ind.layerToChip);
}

void GA::mutate_segmentation(std::vector<int> &seg)
{
    const int flip_prob = 5;   // Bit flip probability
    const int shift_prob = 10; // Segment shift probability

    // Bit flip: insert or delete breakpoint
    for (size_t i = 0; i < seg.size(); ++i)
    {
        if (ThreadSafeRandom::rand_int(0, 99) < flip_prob)
        {
            seg[i] ^= 1;
        }
    }

    // Segment shift: move existing breakpoint left or right
    for (size_t i = 0; i < seg.size(); ++i)
    {
        if (seg[i] == 1 && ThreadSafeRandom::rand_int(0, 99) < shift_prob)
        {
            int direction = ThreadSafeRandom::rand_int(0, 1); // 0: left, 1: right
            if (direction == 0 && i > 0 && seg[i - 1] == 0)
            {
                seg[i] = 0;
                seg[i - 1] = 1;
            }
            else if (direction == 1 && i + 1 < seg.size() && seg[i + 1] == 0)
            {
                seg[i] = 0;
                seg[i + 1] = 1;
            }
        }
    }
}

void GA::mutate_mapping(const std::vector<int> &segmentation, std::vector<std::vector<int>> &mapping)
{
    double progress = static_cast<double>(current_generation) / generations;

    // Probability: sorted by disturbance level from small to large
    // ① Replace mapping[b][l] with new chiplet (ensure space traversability)
    int PROB_RANDOM_REPLACEMENT;
    // ② Swap adjacent layers
    int PROB_SWAP_ADJACENT_LAYERS;
    // ③ Swap batches on same layer
    int PROB_SWAP_BATCHES_ON_LAYER;
    // ④ Random reordering within segment
    int PROB_SHUFFLE_SEGMENT;
    // ⑤ Segment remapping
    int PROB_REMAPPING_SEGMENT;
    // ⑥ Swap two segments
    int PROB_SWAP_SEGMENTS;
    // ⑦ Swap two batches
    int PROB_SWAP_BATCHES;

    if (progress < 0.3)
    {
        PROB_RANDOM_REPLACEMENT = 10;
        PROB_SWAP_ADJACENT_LAYERS = 5;
        PROB_SWAP_BATCHES_ON_LAYER = 5;
        PROB_SHUFFLE_SEGMENT = 10;
        PROB_REMAPPING_SEGMENT = 20;
        PROB_SWAP_SEGMENTS = 20;
        PROB_SWAP_BATCHES = 20;
    }
    else if (progress < 0.7)
    {
        PROB_RANDOM_REPLACEMENT = 10;
        PROB_SWAP_ADJACENT_LAYERS = 5;
        PROB_SWAP_BATCHES_ON_LAYER = 5;
        PROB_SHUFFLE_SEGMENT = 10;
        PROB_REMAPPING_SEGMENT = 10;
        PROB_SWAP_SEGMENTS = 10;
        PROB_SWAP_BATCHES = 10;
    }
    else
    {
        PROB_RANDOM_REPLACEMENT = 10;
        PROB_SWAP_ADJACENT_LAYERS = 10;
        PROB_SWAP_BATCHES_ON_LAYER = 10;
        PROB_SHUFFLE_SEGMENT = 3;
        PROB_REMAPPING_SEGMENT = 3;
        PROB_SWAP_SEGMENTS = 2;
        PROB_SWAP_BATCHES = 2;
    }

    // ① replace mapping[b][l] to new chiplet
    if (ThreadSafeRandom::rand_int(0, 99) < PROB_RANDOM_REPLACEMENT){
        auto b=ThreadSafeRandom::rand_int(0, BATCH_SIZE-1);
        auto l=ThreadSafeRandom::rand_int(0, LAYER_NUM-1);
        mapping[b][l] = ThreadSafeRandom::rand_int(0, CHIPLET_NUM - 1);
    }

    // ② swap layer
    if (ThreadSafeRandom::rand_int(0, 99) < PROB_SWAP_ADJACENT_LAYERS && LAYER_NUM > 1){
        auto b=ThreadSafeRandom::rand_int(0, BATCH_SIZE-1);
        auto l=ThreadSafeRandom::rand_int(0, LAYER_NUM-2);
        std::swap(mapping[b][l], mapping[b][l + 1]);
    }

    // ③ swap across batch
    if (ThreadSafeRandom::rand_int(0, 99) < PROB_SWAP_BATCHES_ON_LAYER)
    {
        auto l= ThreadSafeRandom::rand_int(0, LAYER_NUM - 1);
        int b1 = ThreadSafeRandom::rand_int(0, BATCH_SIZE - 1);
        int b2 = ThreadSafeRandom::rand_int(0, BATCH_SIZE - 1);
        if(b1 != b2)
            std::swap(mapping[b1][l], mapping[b2][l]);
    }


    // get segments
    std::vector<std::pair<int, int>> segments;
    int start = 0;
    for (size_t i = 0; i < segmentation.size(); ++i)
    {
        if (segmentation[i] == 1)
        {
            segments.emplace_back(start, i + 1);
            start = i + 1;
        }
    }
    segments.emplace_back(start, LAYER_NUM);

    // ④ segment sfuffle
    if (!segments.empty() && ThreadSafeRandom::rand_int(0, 99) < PROB_SHUFFLE_SEGMENT)
    {
        auto seg_index= ThreadSafeRandom::rand_int(0, segments.size() - 1);
        auto [a, b] = segments[seg_index];
        auto bb= ThreadSafeRandom::rand_int(0, BATCH_SIZE - 1);
        std::vector<int> temp(mapping[bb].begin() + a, mapping[bb].begin() + b);
        std::shuffle(temp.begin(), temp.end(), ThreadSafeRandom::get_generator());
        std::copy(temp.begin(), temp.end(), mapping[bb].begin() + a);
    }

    // ⑤ segment remapping
    if (!segments.empty() && ThreadSafeRandom::rand_int(0, 99) < PROB_REMAPPING_SEGMENT)
    {
        int s = ThreadSafeRandom::rand_int(0, segments.size() - 1);
        auto [a, b] = segments[s];
        int bb = ThreadSafeRandom::rand_int(0, BATCH_SIZE - 1);
        for (int l = a; l < b; ++l)
            mapping[bb][l] = ThreadSafeRandom::rand_int(0, CHIPLET_NUM - 1);
    }

    // ⑥ swap two segment
    if (segments.size() >= 2 && ThreadSafeRandom::rand_int(0, 99) < PROB_SWAP_SEGMENTS)
    {
        int s1 = ThreadSafeRandom::rand_int(0, segments.size() - 1);
        int s2 = ThreadSafeRandom::rand_int(0, segments.size() - 1);
        if (s1 != s2)
        {
            auto [a1, b1] = segments[s1];
            auto [a2, b2] = segments[s2];
            int len = std::min(b1 - a1, b2 - a2);
            for (int b = 0; b < BATCH_SIZE; ++b)
                for (int i = 0; i < len; ++i)
                    std::swap(mapping[b][a1 + i], mapping[b][a2 + i]);
        }
    }
    // ⑦ swap two batch
    if (BATCH_SIZE >= 2 && ThreadSafeRandom::rand_int(0, 99) < PROB_SWAP_BATCHES)
    {
        int b1 = ThreadSafeRandom::rand_int(0, BATCH_SIZE - 1);
        int b2 = ThreadSafeRandom::rand_int(0, BATCH_SIZE - 1);
        if (b1 != b2)
        {
            std::swap(mapping[b1], mapping[b2]);
        }
    }
}

void GA::crossover_and_mutate_parallel()
{
    size_t total = new_population.size();
    size_t index = 1;

    while (index < total)
    {
        std::vector<std::thread> threads;

        size_t batch_size = std::min(thread_num, static_cast<unsigned int>(total - index));
        for (size_t i = 0; i < batch_size; ++i)
        {
            threads.push_back(std::thread(&GA::crossover_and_mutate, this, index));
            index++;
        }

        for (auto &t : threads)
        {
            t.join();
        }
    }
    // for (size_t i = 0; i < new_population.size(); ++i) {
    //     GA::crossover_and_mutate(i);
    // }
}

void GA::crossover_and_mutate(size_t i)
{
    Individual parent1 = tournament_selection();
    Individual parent2 = tournament_selection();
    Individual child = crossover(parent1, parent2);
    mutate(child);
    // std::lock_guard<std::mutex> guard(mtx);
    new_population[i] = child; //
}

Individual GA::tournament_selection()
{
    std::vector<Individual> tournament;
    for (int i = 0; i < 5; ++i)
    {
        tournament.push_back(population[ThreadSafeRandom::rand_int(0, pop_size - 1)]);
    }
    std::sort(tournament.begin(), tournament.end(),
              [](const Individual &a, const Individual &b)
              { return a.fitness > b.fitness; });
    return tournament[0];
}

void GA::update_best_solution(std::vector<Individual> &population)
{
    for (const auto &ind : population)
    {
        process_latency.push_back(ind.latency);
        process_energy.push_back(ind.energy);
        if (ind.fitness > best_fitness)
        {
            best_solution = ind;
            best_fitness = ind.fitness;
        }
    }
}

void GA::run()
{
    best_fitness = -1e9;
    current_generation = 1;
    process_latency.clear();
    process_energy.clear();
    initialize_population();
    update_best_solution(population);
    for (int _ : tqdm(generations - 1, "GA: run"))
    {
        (void)_;
        std::sort(population.begin(), population.end(),
                  [](const Individual &a, const Individual &b)
                  { return a.fitness > b.fitness; });
        // std::cout << "Generation " << gen << " best fitness: " << population[0].fitness << "\n";
        new_population.clear();
        new_population.resize(pop_size);
        new_population[0] = population[0];            // save best
        crossover_and_mutate_parallel();              
        evaluate_population_parallel(new_population); 
        population = new_population;
        update_best_solution(population);
        current_generation++;
    }
}

void GA::random_run()
{
    std::cout << "GA: random run" << std::endl;
    population.clear();
    process_latency.clear();
    process_energy.clear();
    population.resize(pop_size * generations);
    for (size_t i = 0; i < population.size(); i++)
    {
        auto [seg, laytochip] = random_mapping(BATCH_SIZE, LAYER_NUM, CHIPLET_NUM);
       population[i] = Individual(seg, laytochip);
    }
    std::cout << "init finished" << std::endl;
    evaluate_population_parallel(population); 
    std::cout << "eva finished" << std::endl;
    best_fitness = -1e9;
    update_best_solution(population);
    std::cout << "random run finished" << std::endl;
}

std::tuple<cycle_t, energy_t, double, mc_t> GA::get_best_res()
{
    

    size_t total = engines.size();
    size_t index = 0;

    double total_latency = 0;
    energy_t total_energy = 0;
    double total_edp = 0;

    while (index < total)
    {
        std::vector<std::future<std::tuple<cycle_t, energy_t>>> futures;

        size_t batch_size = std::min(thread_num, static_cast<unsigned int>(total - index));
        for (size_t i = 0; i < batch_size; ++i)
        {
            auto engine= engines[i][0].get();
            futures.push_back(std::async(std::launch::async, [engine,segmentation = best_solution.segmentation, layerMap = best_solution.layerToChip]() {
                    engine->setSegmentation(segmentation, layerMap);
                    auto [latency, energy] = engine->calcLatencyAndEnergy();
                    return std::make_tuple(latency, energy);
                }));
            index++;
        }

        for (auto &f : futures)
        {
            auto [latency, energy] = f.get();
            total_latency += latency;
            total_energy += energy;
            total_edp+=latency*energy;
        }
    }
    auto mc = engines.back()[0]->calcMonetaryCost();
    return {total_latency/engines.size(), total_energy/engines.size(),total_edp/engines.size(), mc};
}



void GA::save_best_solution(const std::string &filename,int micro_batch_size)
{
    nlohmann::json j;
    j["segmentation"] = best_solution.segmentation;
    j["layer_to_chip"] = best_solution.layerToChip;
    j["micro_batch_size"] = micro_batch_size;
    if(!engines.empty() && !engines.back().empty() && !engines.back()[0]->batchedModels.empty()){
        auto model = engines.back()[0]->batchedModels[0];
        j["mapping_layers"] = nlohmann::json::array();
        for(size_t mapping_id = 0; mapping_id < model->mapping_len(); ++mapping_id){
            const auto& mapping_node = model->getMappingNode(static_cast<Network::mapping_id_t>(mapping_id));
            nlohmann::json item;
            item["mappingNodeID"] = mapping_id;
            item["mappingNodeName"] = mapping_node.name;
            item["execLayerIDs"] = mapping_node.exec_layer_ids;
            item["execLayerNames"] = nlohmann::json::array();
            for(auto exec_id : mapping_node.exec_layer_ids){
                item["execLayerNames"].push_back(model->getNode(exec_id).name());
            }
            j["mapping_layers"].push_back(item);
        }
    }
    std::ofstream o(filename);
    o << std::setw(4) << j << std::endl;
    std::cout << "Best solution saved to " << filename << "\n";
    std::cout << "Best fitness: " << best_fitness << "\n";
    std::cout << "Latency: " << best_solution.latency << ", Energy: " << best_solution.energy << ", EDP: " << 1/best_fitness << "\n";
}

void GA::save_latency_detail(const std::string &filename, const std::string& stats_mode)
{
    nlohmann::json j;
    engines.back()[0]->setSegmentation(best_solution.segmentation, best_solution.layerToChip);
    engines.back()[0]->calcLatencyAndEnergy();
    j=engines.back()[0]->get_latency_detail(stats_mode);
    std::ofstream o(filename);
    o << std::setw(4) << j << std::endl;
    std::cout << "Best solution latency detail saved to " << filename << "\n";
}

void GA::save_energy_detail(const std::string &filename, const std::string& stats_mode)
{
    nlohmann::json j;
    engines.back()[0]->setSegmentation(best_solution.segmentation, best_solution.layerToChip);
    engines.back()[0]->calcLatencyAndEnergy();
    j=engines.back()[0]->get_energy_detail(stats_mode);
    std::ofstream o(filename);
    o << std::setw(4) << j << std::endl;
    std::cout << "Best solution energy detail saved to " << filename << "\n";
}

void GA::save_mc_detail(const std::string &filename)
{
    nlohmann::json j;
    engines.back()[0]->calcMonetaryCost();
    j=engines.back()[0]->get_mc_detail();
    std::ofstream o(filename);
    o << std::setw(4) << j << std::endl;
    std::cout << "Best solution mc detail saved to " << filename << "\n";
}

void GA::save_progress(const std::string &filename)
{

    rapidcsv::Document doc;
    doc.SetColumnName(0, "latency");
    doc.SetColumnName(1, "energy");
    doc.SetColumn<cycle_t>("latency", process_latency);
    doc.SetColumn<energy_t>("energy", process_energy);
    doc.Save(filename);
    return;
}
