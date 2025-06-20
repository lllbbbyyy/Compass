#include <iostream>
#include <memory>
#include <iomanip>
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

using namespace std;

double ccost_func(cycle_t l,energy_t e,mc_t mc,double mc_limit=49.14685){
	return l*e*(max(mc,mc_limit)/mc_limit);
}
int main()
{
	// std::cout << std::fixed << std::setprecision(2);
	ThreadSafeRandom::set_seed(42); // Set a fixed seed for reproducibility

	ReqGenerator::inputLengthsFile="./config/govreport_input_token_lens.json";
	ReqGenerator::outputLengthsFile="./config/govreport_output_token_lens.json";

	constexpr double baseline=1.19e17;
	int init_chip_number = 4;
	int decode_chip_num=3;

	int batch_size = 66;
	int micro_batch_size = 2;
	
	int nop_bw=32;
	int dram_bw=64;
	

	vector<shared_ptr<CoreMapper>> chips;
	chips.emplace_back(createPolarCoreMapper(32768*64,2048 KB));
	chips.emplace_back(createPolarCoreMapper(256,64 KB));
	chips.emplace_back(createPolarCoreMapper(256,64 KB));
	chips.emplace_back(createPolarCoreMapper(256,64 KB));

	DEBUG("chips created");

	auto [ylen, xlen] = closest_factors(init_chip_number);
	auto noc = createNoC(xlen, ylen, nop_bw, dram_bw, 4);
	DEBUG("noc created");

	ReqGenerator generator(batch_size);
	vector<shared_ptr<Network>> batched_models;
	auto batches = generator.generateReq(micro_batch_size,2,64);
	for (int i : tqdm(batch_size / micro_batch_size))
	{
		DEBUG("batch", i);
		for (auto &req : batches[i])
		{
			DEBUG("	", req);
		}
	}
	// exit(0);
	for (int i : tqdm(batch_size / micro_batch_size))
	{
		// auto n=create_GPT3(batches[i],32,4096,32,128);
		auto n = create_GPT3(batches[i], 1, 4096, 32, 128, 2048);
		// auto n =gen_convs(36);
		// auto n=create_GPT3(batches[i],1,256,8,32);
		batched_models.emplace_back(n);
	}
	size_t layer_lens = batched_models[0]->len();
	for (size_t i = 0; i < layer_lens; i++)
	{
		DEBUG("layer", i, batched_models[0]->getNode(i).name());
	}
	DEBUG("models created", layer_lens);


	vector<int> segmentation;
	vector<vector<cidx_t>> layerToChip(batch_size/micro_batch_size, vector<cidx_t>());
	for (size_t i = 0; i < layer_lens; i++)
	{
		layerToChip[0].push_back(0);
		if (i != layer_lens - 1)
		{
			if ((i + 1) % decode_chip_num == 0)
				segmentation.push_back(1);
			else
				segmentation.push_back(0);
		}
		for(size_t j=0;j<layerToChip.size()-1;j++)
		{
			layerToChip[j+1].push_back((i % decode_chip_num)+1);
		}
	}
	DEBUG("segmentation and layerToChip created");

	auto modelEngine = CompassModelEngine(batched_models, chips, noc, segmentation, layerToChip);

	auto [latency, energy] = modelEngine.calcLatencyAndEnergy();
	auto mc = modelEngine.calcMonetaryCost();

	cout << "test1" << endl;
	cout << "latency: " << latency << " energy: " << energy << " mc: " << mc << endl;

	cout<<"cost: "<<ccost_func(latency, energy, mc) << endl;
	auto mc_detail=modelEngine.get_mc_detail();
	cout<<"cost detail"<<mc_detail<<endl;

	auto j=modelEngine.get_latency_detail();

	string filename="./tmp/detail_latency.json";
    std::ofstream o(filename);
    o << std::setw(4) << j << std::endl;
    std::cout << "Best solution latency detail saved to " << filename << "\n";

	return 0;
}