#include <iostream>
#include <memory>
#include <future>
#include <iomanip>
#include <unistd.h>
#include <limits.h>
#include <stdexcept>
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

static len_t get_motivation_gemm_dim(const json& model_info, const string& layer_name, const string& dim_name)
{
	if(!model_info.contains(layer_name)){
		throw std::logic_error("motivation_two_layer model_info missing " + layer_name);
	}
	const auto& layer_info = model_info[layer_name];
	if(!layer_info.contains(dim_name)){
		throw std::logic_error("motivation_two_layer " + layer_name + " missing " + dim_name);
	}
	return layer_info[dim_name].get<len_t>();
}

static len_t get_motivation_layer_dim_or(const json& model_info, const string& layer_name, const string& dim_name, len_t default_value)
{
	if(!model_info.contains(layer_name)){
		return default_value;
	}
	const auto& layer_info = model_info[layer_name];
	if(!layer_info.contains(dim_name)){
		return default_value;
	}
	return layer_info[dim_name].get<len_t>();
}

static void apply_tensor_parallel_to_model_info(json& model_info, len_t tensor_parall)
{
	if(tensor_parall <= 0){
		throw std::logic_error("tensor_parall must be positive.");
	}
	len_t d_model = model_info["d_model"].get<len_t>();
	len_t d_ffn = model_info["d_ffn"].get<len_t>();
	if(d_model % tensor_parall != 0 || d_ffn % tensor_parall != 0){
		throw std::logic_error("tensor_parall must divide both d_model and d_ffn.");
	}
	model_info["tensor_parall"] = tensor_parall;
}

static std::pair<len_t, len_t> get_model_tiling_sizes(const json& model_info, len_t d_model, len_t d_ffn)
{
	len_t d_model_tiling_size = d_model;
	len_t d_ffn_tiling_size = d_ffn;
	if(model_info.contains("tensor_parall")){
		len_t tensor_parall = model_info["tensor_parall"].get<len_t>();
		if(tensor_parall <= 0 || d_model % tensor_parall != 0 || d_ffn % tensor_parall != 0){
			throw std::logic_error("model_info.tensor_parall must divide both d_model and d_ffn.");
		}
		d_model_tiling_size = d_model / tensor_parall;
		d_ffn_tiling_size = d_ffn / tensor_parall;
	}
	if(model_info.contains("d_model_tiling_size")){
		d_model_tiling_size = model_info["d_model_tiling_size"].get<len_t>();
	}
	if(model_info.contains("d_ffn_tiling_size")){
		d_ffn_tiling_size = model_info["d_ffn_tiling_size"].get<len_t>();
	}
	if(d_model_tiling_size <= 0 || d_ffn_tiling_size <= 0){
		throw std::logic_error("Model tiling sizes must be positive.");
	}
	if(d_model % d_model_tiling_size != 0 || d_ffn % d_ffn_tiling_size != 0){
		throw std::logic_error("Model tiling sizes must divide d_model and d_ffn.");
	}
	return {d_model_tiling_size, d_ffn_tiling_size};
}

std::shared_ptr<Network> create_llm(const json& j,const std::vector<Req> &reqs){
	string type= j["type"];
	if(type=="motivation_two_layer"){
		(void)reqs;
		return create_motivation_two_layer(
			get_motivation_gemm_dim(j, "layer_a", "M"),
			get_motivation_gemm_dim(j, "layer_a", "K"),
			get_motivation_gemm_dim(j, "layer_a", "N"),
			get_motivation_gemm_dim(j, "layer_b", "M"),
			get_motivation_gemm_dim(j, "layer_b", "K"),
			get_motivation_gemm_dim(j, "layer_b", "N"),
			get_motivation_layer_dim_or(j, "layer_b", "heads", 1));
	}
	len_t n_layers = j["n_layer"];
	len_t d_model = j["d_model"];
	len_t n_head = j["n_head"];
	len_t d_head = j["d_head"];
	len_t d_ffn = j["d_ffn"];
	auto [d_model_tiling_size, d_ffn_tiling_size] = get_model_tiling_sizes(j, d_model, d_ffn);
	len_t tensor_parallel = d_model/d_model_tiling_size;
	if(j.contains("tensor_parall")){
		tensor_parallel = j["tensor_parall"].get<len_t>();
	}
	const string qkv_projection_mode = j.value("qkv_projection_mode", "separate");
	string mapping_merge_mode;
	if(j.contains("mapping_merge_mode")){
		mapping_merge_mode = j["mapping_merge_mode"].get<string>();
	}
	else if(j.contains("merge_granularity")){
		mapping_merge_mode = j["merge_granularity"].get<string>();
	}
	else if(type=="gpt3_merged" || type=="llama3_merged"){
		mapping_merge_mode = "stage";
	}
	else{
		mapping_merge_mode = "none";
	}
	DEBUG("mapping_merge_mode", mapping_merge_mode);
	DEBUG("qkv_projection_mode", qkv_projection_mode);
	if (type=="llama3"){
		len_t n_kv_heads = j["n_kv_head"];
		return create_llama3(reqs, n_layers, d_model, n_head, d_head, n_kv_heads, d_ffn, d_model_tiling_size, d_ffn_tiling_size, mapping_merge_mode);
	}
	else if(type=="gpt3"){
		return create_GPT3(reqs, n_layers, d_model, n_head, d_head, d_ffn, d_model_tiling_size, d_ffn_tiling_size, mapping_merge_mode);
	}
	else if(type=="gpt3_merged"){
		return create_GPT3_merged(reqs, n_layers, d_model, n_head, d_head, d_ffn, mapping_merge_mode, d_model_tiling_size, d_ffn_tiling_size, qkv_projection_mode, tensor_parallel);
	}
	else if(type=="llama3_merged"){
		len_t n_kv_heads = j["n_kv_head"];
		return create_llama3_merged(reqs, n_layers, d_model, n_head, d_head, n_kv_heads, d_ffn, mapping_merge_mode, d_model_tiling_size, d_ffn_tiling_size, qkv_projection_mode, tensor_parallel);
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
		std::cerr << "Need " <<3<<" arguments, but got "<<argc-1<<std::endl;
		return 1;
	}

	std::string config_filename = argv[1];
	std::string input_filename = argv[2];
	std::string output_filename = argv[3];

	// 2. open file
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

	// 2. open file
	std::ifstream inFile(input_filename);
	if (!inFile.is_open())
	{
		std::cerr << "Error: cannot open input file " << input_filename << std::endl;
		return 1;
	}

	json j;
	inFile >> j;

	// read config
	int init_chip_number = j["num_chiplets"];
	int batch_size = config_j["batch_size"];
	int micro_batch_size = 1;
	if(config_j.contains("micro_batch")){
		micro_batch_size = config_j["micro_batch"];
	}
	else if(j.contains("micro_batch")){
		micro_batch_size = j["micro_batch"];
	}
	if(j.contains("tensor_parall")){
		int tensor_parall=j["tensor_parall"];
		apply_tensor_parallel_to_model_info(config_j["model_info"], tensor_parall);
		DEBUG("tensor_parall: ", tensor_parall);
	}

	auto &chips_info = j["chiplets"];
	assert((int)chips_info.size() == init_chip_number);

	vector<shared_ptr<CoreMapper>> chips;
	for (const auto &chip_info : chips_info)
	{
		string chip_type = chip_info["type"];
		vol_t buffer_size = chip_info["buffer_size"];
		buffer_size = buffer_size KB;
		int compute_units = chip_info["compute_units"];
		string macs="";
		if(chip_info.contains("macs")){
			macs=chip_info["macs"];
		}
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
			chips.emplace_back(createPolarCoreMapper(compute_units, buffer_size,chip_type,macs));
		}
	}

	DEBUG("chips created");

	int chip_x, chip_y;
	if(j.contains("chip_x")&&j.contains("chip_y")){
		chip_x=j["chip_x"];
		chip_y=j["chip_y"];
		assert(chip_x*chip_y==init_chip_number);
	}
	else{
		tie(chip_x, chip_y) = closest_factors(init_chip_number);
	}

	DEBUG("chiplet number", init_chip_number, chip_x, chip_y);
	bw_t nop_bw = j["nop_bw"];
	bw_t dram_bw = j["dram_bw"];
	int dram_num = config_j["dram_num"];
	auto noc = createNoC(chip_x, chip_y, nop_bw, dram_bw, dram_num);
	DEBUG("noc created");

	ReqGenerator::inputLengthsFile = config_j["req_generator_input_length_path"];
	ReqGenerator::outputLengthsFile = config_j["req_generator_output_length_path"];
	int req_number= config_j["req_number"];
	bool is_chunked_prefill=config_j["is_chunked_prefill"];
	ReqGenerator generator(batch_size);
	vector<vector<shared_ptr<Network>>> batched_models(req_number);

	string req_gen_mode = config_j["req_gen_mode"];
	int num_prefill = config_j["req_prefill_number"];
	int num_decode = config_j["req_decode_number"];

	auto model_info = config_j["model_info"];
	string model_type = model_info["type"];
	int chunked_size= 1;
	if(is_chunked_prefill){
		if(config_j.contains("chunked_prefill_size")){
			chunked_size=config_j["chunked_prefill_size"];
			if(chunked_size<=0){
				throw std::logic_error("chunked_prefill_size must be positive");
			}
		}
		else{
			chunked_size=generator.getNextInputLength()/req_number;
		}
		DEBUG("chunked_prefill_size", chunked_size);
	}
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
		if(is_chunked_prefill){
			auto chunked_prefill_req=Req(0, Req::Type::ChunkedPrefill, chunked_size, j*chunked_size);
			batches[0][0]=chunked_prefill_req;
		}
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
			if(model_type=="gpt3"||model_type=="gpt3_merged"||model_type=="llama3"||model_type=="llama3_merged"||model_type=="motivation_two_layer"){
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
	size_t mapping_layer_lens = batched_models[0][0]->mapping_len();
	for (size_t i = 0; i < mapping_layer_lens; i++)
	{
		const auto& mapping_node = batched_models[0][0]->getMappingNode(static_cast<Network::mapping_id_t>(i));
		lid_t first_exec = mapping_node.exec_layer_ids.empty() ? -1 : mapping_node.exec_layer_ids.front();
		lid_t last_exec = mapping_node.exec_layer_ids.empty() ? -1 : mapping_node.exec_layer_ids.back();
		DEBUG("mapping layer", i, mapping_node.name, "exec_count", mapping_node.exec_layer_ids.size(), "exec_range", first_exec, last_exec);
	}
	DEBUG("mapping layers created", mapping_layer_lens);

	string run_mode= config_j["run_mode"];
	string best_solution_file=config_j["best_mapping_save_path"];
	string detail_latency_file=config_j["detail_latency_save_path"];
	string detail_energy_file=config_j["detail_energy_save_path"];
	string detail_mc_file=config_j["detail_mc_save_path"];
	string detail_stats_mode = "layer";
	if(config_j.contains("detail_stats_mode")){
		detail_stats_mode = config_j["detail_stats_mode"].get<string>();
	}
	else if(config_j.contains("detail_stat_mode")){
		detail_stats_mode = config_j["detail_stat_mode"].get<string>();
	}
	DEBUG("detail_stats_mode", detail_stats_mode);
	string search_process_file=config_j["search_process_save_path"];
	string exec_load_file=config_j["exec_load_path"];

	GA::pop_size = config_j["GA_population_size"];
	GA::generations = config_j["GA_generations"];

	cycle_t latency;
	energy_t energy;
	double edp_res;
	mc_t mc;

	if(run_mode=="GA"){
		auto ga_engine = GA(batched_models, chips, noc);
		ga_engine.run();
		auto [l, e, edp, m] = ga_engine.get_best_res();
		if(!best_solution_file.empty())
			ga_engine.save_best_solution(best_solution_file, micro_batch_size);
		if(!detail_latency_file.empty())
			ga_engine.save_latency_detail(detail_latency_file, detail_stats_mode);
		if(!detail_energy_file.empty())
			ga_engine.save_energy_detail(detail_energy_file, detail_stats_mode);
		if(!detail_mc_file.empty())
			ga_engine.save_mc_detail(detail_mc_file);
		if(!search_process_file.empty())
			ga_engine.save_progress(search_process_file);
		latency = l;
		energy = e;
		edp_res=edp;
		mc = m;
	}
	else if(run_mode=="random"){
		auto ga_engine = GA(batched_models, chips, noc);
		ga_engine.random_run();
		auto [l, e, edp, m] = ga_engine.get_best_res();
		if(!best_solution_file.empty())
			ga_engine.save_best_solution(best_solution_file, micro_batch_size);
		if(!detail_latency_file.empty())
			ga_engine.save_latency_detail(detail_latency_file, detail_stats_mode);
		if(!detail_energy_file.empty())
			ga_engine.save_energy_detail(detail_energy_file, detail_stats_mode);
		if(!detail_mc_file.empty())
			ga_engine.save_mc_detail(detail_mc_file);
		if(!search_process_file.empty())
			ga_engine.save_progress(search_process_file);
		latency = l;
		energy = e;
		edp_res=edp;
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
		double total_edp = 0;
		vector<cycle_t> latencys;
		vector<energy_t> energys;
		vector<double> edps;
		auto thread_num=std::thread::hardware_concurrency();
		std::vector<std::unique_ptr<CompassModelEngine>>engines;
		engines.reserve(batched_models.size());
		for (size_t i = 0; i < batched_models.size(); ++i) {
			engines.push_back(std::make_unique<CompassModelEngine>(batched_models[i], chips, noc, segmentation, layerToChip));
		}

		while (index < total)
		{
			std::vector<std::future<std::tuple<cycle_t, energy_t>>> futures;

			size_t parall_size = std::min(thread_num, static_cast<unsigned int>(total - index));
			for (size_t i = 0; i < parall_size; ++i)
			{
				size_t current_index = index;
				futures.push_back(std::async(std::launch::async, [&engines,current_index]() {
						auto [l,e]=engines[current_index]->calcLatencyAndEnergy();
						return std::make_tuple(l,e);
					}));
				index++;
			}

			for (auto &f : futures)
			{
				auto [latency, energy] = f.get();
				total_latency += latency;
				total_energy += energy;
				total_edp+=latency*energy;
				latencys.push_back(latency);
				energys.push_back(energy);
				edps.push_back(latency*energy);
			}
		}
		DEBUG("exec total res",total_latency, total_energy, total_edp);
		CompassModelEngine model_engine(batched_models.back(), chips, noc, segmentation, layerToChip);
		auto m=model_engine.calcMonetaryCost();
		latency=total_latency/total;
		energy=total_energy/total;
		edp_res=total_edp/total;
		mc=m;
		DEBUG("exec avg res",latency, energy, edp_res, mc);
		model_engine.calcLatencyAndEnergy(); //for detail
		if(!detail_latency_file.empty()){
			auto j=model_engine.get_latency_detail(detail_stats_mode);
			std::ofstream o(detail_latency_file);
			o << std::setw(4) << j << std::endl;
			std::cout << "Best mapping latency detail saved to " << detail_latency_file << "\n";
		}
		if(!detail_energy_file.empty()){
			auto j=model_engine.get_energy_detail(detail_stats_mode);
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
		doc.SetColumnName(2, "edp");
		doc.SetColumnName(3, "mc");
		doc.SetColumn<cycle_t>("latency", latencys);
		doc.SetColumn<energy_t>("energy", energys);
		doc.SetColumn<double>("edp", edps);
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
		doc.SetColumnName(2, "edp");
		doc.SetColumnName(3, "mc");
		doc.SetColumn<cycle_t>("latency", vector<cycle_t>{latency});
		doc.SetColumn<energy_t>("energy", vector<energy_t>{energy});
		doc.SetColumn<energy_t>("edp", vector<double>{edp_res});
		doc.SetColumn<mc_t>("mc", vector<mc_t>{mc});

		doc.Save(output_filename);
	}
	return 0;
}
