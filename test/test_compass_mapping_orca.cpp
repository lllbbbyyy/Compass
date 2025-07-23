#include <iostream>
#include <memory>
#include <future>
#include <iomanip>
#include <unistd.h>
#include <limits.h>
#include "layer_engine.h"
#include "core.h"
#include "nns/nns.h"
#include "noc.h"
#include "debug.h"
#include "compass/request_generator.h"
#include "nns/nns.h"
#include "model_engine.h"
#include "compass/utils_compass.h"
#include "tqdm/tqdm.h"
#include "json.hpp"
#include "compass/GA.h"
#include "rapidcsv.h"

using namespace std;
using json = nlohmann::json;

std::shared_ptr<Network> create_llm(const json& j,const std::vector<Req> &reqs){
	len_t n_layers = j["n_layer"];
	len_t d_model = j["d_model"];
	len_t n_head = j["n_head"];
	len_t d_head = j["d_head"];
	len_t d_ffn = j["d_ffn"];
	len_t d_model_tiling_size= j["d_model_tiling_size"];
	len_t d_ffn_tiling_size= j["d_ffn_tiling_size"];
	string type= j["type"];
	if (type=="llama3"){
		len_t n_kv_heads = j["n_kv_head"];
		return create_llama3(reqs, n_layers, d_model, n_head, d_head, n_kv_heads, d_ffn, d_model_tiling_size, d_ffn_tiling_size);
	}
	else if(type=="gpt3"){
		return create_GPT3(reqs, n_layers, d_model, n_head, d_head, d_ffn, d_model_tiling_size, d_ffn_tiling_size);
	}
	return nullptr;
}

int main(int argc, char *argv[])
{
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current directory: " << cwd << std::endl;
    } else {
        perror("getcwd() error");
    }

	std::cout << std::fixed << std::setprecision(2);
	if (argc != 4)
	{
		std::cerr << "Usage: " << argv[0] << " <config_json_file> <input_json_file> <res_csv_file>" << std::endl;
		return 1;
	}

	std::string config_filename = argv[1];
	std::string input_filename = argv[2];
	std::string output_filename = argv[3];

	// 2. 打开文件
	std::ifstream configFile(config_filename);
	if (!configFile.is_open())
	{
		std::cerr << "Error: cannot open config file " << config_filename << std::endl;
		return 1;
	}

	json config_j;
	configFile >> config_j;

	int seed=config_j["seed"];
	ThreadSafeRandom::set_seed(seed);

	// 2. 打开文件
	std::ifstream inFile(input_filename);
	if (!inFile.is_open())
	{
		std::cerr << "Error: cannot open input file " << input_filename << std::endl;
		return 1;
	}

	json j;
	inFile >> j;

	// read config
	size_t init_chip_number = j["num_chiplets"];
	int batch_size = config_j["batch_size"];
	int micro_batch_size = j["micro_batch"];

	auto &chips_info = j["chiplets"];
	assert(chips_info.size() == init_chip_number);

	vector<shared_ptr<CoreMapper>> chips;
	for (const auto &chip_info : chips_info)
	{
		string chip_type = chip_info["type"];
		vol_t buffer_size = chip_info["buffer_size"];
		buffer_size = buffer_size KB;
		int compute_units = chip_info["compute_units"];
		if (chip_type == "NVDLA")
		{
			chips.emplace_back(createPolarCoreMapper(compute_units, buffer_size));
		}
		else if (chip_type == "Eyeriss")
		{
			chips.emplace_back(createEyerissCoreMapper(compute_units, buffer_size));
		}
		else
		{
			assert(0);
		}
	}

	DEBUG("chips created");

	auto [chip_x, chip_y] = closest_factors(init_chip_number);
	DEBUG("chiplet number", init_chip_number, chip_x, chip_y);
	bw_t nop_bw = j["nop_bw"];
	bw_t dram_bw = j["dram_bw"];
	int dram_num = config_j["dram_num"];
	auto noc = createNoC(chip_x, chip_y, nop_bw, dram_bw, dram_num);
	DEBUG("noc created");

	ReqGenerator::inputLengthsFile = config_j["req_generator_input_length_path"];
	ReqGenerator::outputLengthsFile = config_j["req_generator_output_length_path"];
	int req_number= config_j["req_number"];
	ReqGenerator generator(batch_size);
	vector<vector<shared_ptr<Network>>> batched_models(req_number);

	string req_gen_mode = config_j["req_gen_mode"];
	int num_prefill = config_j["req_prefill_number"];
	int num_decode = config_j["req_decode_number"];

	auto model_info = config_j["model_info"];
	string model_type = model_info["type"];
	int prefill_size = 2048*5;
	auto prefill_req=Req(0, Req::Type::Prefill, prefill_size, 0);
	for(int j:tqdm(req_number,"ReqGenerator: generate requests and create model"))
	{
		batchedReqs_t batches;
		if(req_gen_mode=="normal"){
			batches = generator.generateReq(micro_batch_size);
		}
		else if(req_gen_mode=="fixed"){
			batches = generator.generateReq(micro_batch_size, num_prefill, num_decode);
		}
		else{
			assert(0);
		}
		if(j==0)
			batches[batch_size / micro_batch_size-1][micro_batch_size-1]=prefill_req;
		std::shared_ptr<Network> n;
		DEBUG("model",j);
		for (int i=0;i<batch_size / micro_batch_size;i++)
		{
			DEBUG("batch", i);
			for (auto &req : batches[i])
			{
				DEBUG("	", req);
			}
			// auto n=create_GPT3(batches[i],32,4096,32,128);
			
			// auto n =gen_convs(36);
			// auto n=create_GPT3(batches[i],1,256,8,32);
			if(model_type=="gpt3"||model_type=="llama3"){
				n = create_llm(model_info, batches[i]);
			}
			else{
				assert(0);
			}

			batched_models[j].emplace_back(n);
		}
	}


	size_t layer_lens = batched_models[0][0]->len();
	for (size_t i = 0; i < layer_lens; i++)
	{
		DEBUG("layer", i, batched_models[0][0]->getNode(i).name());
	}
	DEBUG("models created", layer_lens);

	string run_mode= config_j["run_mode"];
	string best_solution_file=config_j["best_mapping_save_path"];
	string detail_latency_file=config_j["detail_latency_save_path"];
	string detail_energy_file=config_j["detail_energy_save_path"];
	string detail_mc_file=config_j["detail_mc_save_path"];
	string search_process_file=config_j["search_process_save_path"];
	string exec_load_file=config_j["exec_load_path"];

	GA::pop_size = config_j["GA_population_size"];
	GA::generations = config_j["GA_generations"];

	cycle_t latency;
	energy_t energy;
	mc_t mc;

	if(run_mode=="GA"){
		auto ga_engine = GA(batched_models, chips, noc);
		ga_engine.run();
		auto [l, e, m] = ga_engine.get_best_res();
		if(!best_solution_file.empty())
			ga_engine.save_best_solution(best_solution_file, micro_batch_size);
		if(!detail_latency_file.empty())
			ga_engine.save_latency_detail(detail_latency_file);
		if(!detail_energy_file.empty())
			ga_engine.save_energy_detail(detail_energy_file);
		if(!detail_mc_file.empty())
			ga_engine.save_mc_detail(detail_mc_file);
		if(!search_process_file.empty())
			ga_engine.save_progress(search_process_file);
		latency = l;
		energy = e;
		mc = m;
	}
	else if(run_mode=="random"){
		auto ga_engine = GA(batched_models, chips, noc);
		ga_engine.random_run();
		auto [l, e, m] = ga_engine.get_best_res();
		if(!best_solution_file.empty())
			ga_engine.save_best_solution(best_solution_file, micro_batch_size);
		if(!detail_latency_file.empty())
			ga_engine.save_latency_detail(detail_latency_file);
		if(!detail_energy_file.empty())
			ga_engine.save_energy_detail(detail_energy_file);
		if(!detail_mc_file.empty())
			ga_engine.save_mc_detail(detail_mc_file);
		if(!search_process_file.empty())
			ga_engine.save_progress(search_process_file);
		latency = l;
		energy = e;
		mc = m;
	}
	else if(run_mode=="exec"){
		json exec_j;
		std::ifstream execFile(exec_load_file);
		if (!execFile.is_open())
		{
			std::cerr << "Error: cannot open exec file " << exec_load_file << std::endl;
			return 1;
		}
		execFile >> exec_j;
		std::vector<int> segmentation=exec_j["segmentation"];
		std::vector<std::vector<int>> layerToChip=exec_j["layer_to_chip"];

		size_t total = batched_models.size();
		size_t index = 0;

		double total_latency = 0;
		energy_t total_energy = 0;
		vector<cycle_t> latencys;
		vector<energy_t> energys;
		auto thread_num=std::thread::hardware_concurrency();
		while (index < total)
		{
			std::vector<std::future<std::tuple<cycle_t, energy_t>>> futures;

			size_t parall_size = std::min(thread_num, static_cast<unsigned int>(total - index));
			for (size_t i = 0; i < parall_size; ++i)
			{
				futures.push_back(std::async(std::launch::async, [batched_models,chips, noc, segmentation, layerToChip,index]() {
						CompassModelEngine model_engine(batched_models[index], chips, noc, segmentation, layerToChip);
						auto [l,e]=model_engine.calcLatencyAndEnergy();
						return std::make_tuple(l,e);
					}));
				index++;
			}

			for (auto &f : futures)
			{
				auto [latency, energy] = f.get();
				total_latency += latency;
				total_energy += energy;
				latencys.push_back(latency);
				energys.push_back(energy);
			}
		}

		CompassModelEngine model_engine(batched_models.back(), chips, noc, segmentation, layerToChip);
		auto m=model_engine.calcMonetaryCost();
		latency=total_latency/total;
		energy=total_energy/total;
		mc=m;
		DEBUG("exec avg res",latency, energy, mc);
		model_engine.calcLatencyAndEnergy(); //for detail
		if(!detail_latency_file.empty()){
			auto j=model_engine.get_latency_detail();
			std::ofstream o(detail_latency_file);
			o << std::setw(4) << j << std::endl;
			std::cout << "Best mapping latency detail saved to " << detail_latency_file << "\n";
		}
		if(!detail_energy_file.empty()){
			auto j=model_engine.get_energy_detail();
			std::ofstream o(detail_energy_file);
			o << std::setw(4) << j << std::endl;
			std::cout << "Best mapping energy detail saved to " << detail_energy_file << "\n";
		}
		if(!detail_mc_file.empty()){
			auto j=model_engine.get_mc_detail();
			std::ofstream o(detail_mc_file);
			o << std::setw(4) << j << std::endl;
			std::cout << "Mc detail saved to " << detail_mc_file << "\n";
		}
		rapidcsv::Document doc;
		doc.SetColumnName(0, "latency");
		doc.SetColumnName(1, "energy");
		doc.SetColumnName(2, "mc");
		doc.SetColumn<cycle_t>("latency", latencys);
		doc.SetColumn<energy_t>("energy", energys);
		doc.SetColumn<mc_t>("mc", vector<mc_t>{mc});

		doc.Save(output_filename);
	}
	else{
		assert(0);
	}

	if(run_mode!="exec"){
		rapidcsv::Document doc;
		doc.SetColumnName(0, "latency");
		doc.SetColumnName(1, "energy");
		doc.SetColumnName(2, "mc");
		doc.SetColumn<cycle_t>("latency", vector<cycle_t>{latency});
		doc.SetColumn<energy_t>("energy", vector<energy_t>{energy});
		doc.SetColumn<mc_t>("mc", vector<mc_t>{mc});

		doc.Save(output_filename);
	}
	return 0;
}