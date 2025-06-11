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

int main()
{
	std::cout << std::fixed << std::setprecision(2);

	int init_chip_number = 36;
	int batch_size = 64;
	int micro_batch_size = 1;

	vector<shared_ptr<CoreMapper>> chips;
	for (int i = 0; i < init_chip_number; i++)
	{
		chips.emplace_back(createPolarCoreMapper(1024, 100 MB));
		//chips.emplace_back(createEyerissCoreMapper(1024,100 MB));
	}
	DEBUG("chips created");

	auto noc = createNoC(6, 6, 256, 128, 4);
	DEBUG("noc created");

	ReqGenerator generator(batch_size);
	vector<shared_ptr<Network>> batched_models;
	auto batches = generator.generateReq(micro_batch_size);
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
		auto n = create_GPT3(batches[i], 1, 4096, 32, 128);
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

	///
	vector<shared_ptr<Network>> batched_models2;
	auto batches2 = generator.generateReq(micro_batch_size);
	for (int i : tqdm(batch_size / micro_batch_size))
	{
		// auto n=create_GPT3(batches[i],32,4096,32,128);
		auto n = create_GPT3(batches[i], 1, 4096, 32, 128);
		// auto n =gen_convs(36);
		// auto n=create_GPT3(batches[i],1,256,8,32);
		batched_models2.emplace_back(n);
	}
	///


	vector<int> segmentation;
	vector<cidx_t> layerToChip;
	for (size_t i = 0; i < layer_lens; i++)
	{
		if (i != layer_lens - 1)
		{
			if ((i + 1) % init_chip_number == 0)
				segmentation.push_back(1);
			else
				segmentation.push_back(0);
		}
		layerToChip.push_back(i % init_chip_number);
	}
	DEBUG("segmentation and layerToChip created");

	auto modelEngine = CompassModelEngine(batched_models, chips, noc, segmentation, layerToChip);

	auto [latency, energy] = modelEngine.calcLatencyAndEnergy();
	auto mc = modelEngine.calcMonetaryCost();

	cout << "test1" << endl;
	cout << "latency: " << latency << " energy: " << energy << " mc: " << mc << endl;

	cout << "mc detail:" << endl;
	cout << modelEngine.mcCost << endl;

	auto ga_engine = GA({batched_models,batched_models2}, chips, noc);
	ga_engine.run();
	std::cout<<"saved best latency: "<<ga_engine.best_solution.latency<<" energy: "<<ga_engine.best_solution.energy<<endl;
	auto [l_,e_,mc_]=ga_engine.get_best_res();
	std::cout<<"get best res latency: "<<l_<<" energy: "<<e_<<" mc: "<<mc_<<endl;
	ga_engine.save_latency_detail("tmp/detail_latency.json");
	ga_engine.save_progress("tmp/progress.csv");

	auto ga_edp = ga_engine.best_solution.latency * ga_engine.best_solution.energy;
	std::cout << "speedup " << latency * energy / ga_edp << endl;
	ga_engine.random_run();
	ga_engine.save_latency_detail("tmp/random_best_solution.json");
	ga_engine.save_progress("tmp/random_progress.csv");
	auto random_edp = ga_engine.best_solution.latency * ga_engine.best_solution.energy;
	std::cout << "speedup " << random_edp / ga_edp << endl;
	return 0;
}