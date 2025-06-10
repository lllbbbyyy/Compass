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

	int init_chip_number = 4;
	int batch_size = 4;
	int micro_batch_size = 2;

	vector<shared_ptr<CoreMapper>> chips;
	for (int i = 0; i < init_chip_number; i++)
	{
		chips.emplace_back(createPolarCoreMapper(256, 100 MB));
		// chips.emplace_back(createEyerissCoreMapper(1024,100 MB));
	}
	DEBUG("chips created");

	auto noc = createNoC(2, 2, 256, 128, 2);
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
	// for(int i : tqdm(batch_size/micro_batch_size)){
	// 	//auto n=create_GPT3(batches[i],32,4096,32,128);
	// 	cout<<128*(i+1)<<endl;
	// 	auto n=GEMM4(128*(i+1));
	// 	//auto n =gen_convs(36);
	// 	//auto n=create_GPT3(batches[i],1,256,8,32);
	// 	batched_models.emplace_back(n);
	// }
	auto n1 = GEMM4(128);
	batched_models.emplace_back(n1);
	auto n2 = GEMM4(256);
	batched_models.emplace_back(n2);
	auto n3 = GEMM4(256);
	batched_models.emplace_back(n3);
	auto n4 = GEMM4(384);
	batched_models.emplace_back(n4);

	size_t layer_lens = batched_models[0]->len();
	for (size_t i = 0; i < layer_lens; i++)
	{
		DEBUG("layer", i, batched_models[0]->getNode(i).name());
	}
	DEBUG("models created", layer_lens);

	auto ga_engine = GA(batched_models, chips, noc);
	ga_engine.run();
	CompassLayerEngine::debug_detail = true;
	ga_engine.save_latency_detail("tmp/convs4_best_solution.json");
	ga_engine.save_progress("tmp/conv4_progress.csv");
	cout << "Best solution segmentation: " << endl;
	for (auto i : ga_engine.best_solution.segmentation)
	{
		cout << i << ' ';
	}
	cout << endl;
	cout << "Best solution mapping: " << endl;
	for (auto &v : ga_engine.best_solution.layerToChip)
	{
		for (auto i : v)
		{
			cout << i << ' ';
		}
		cout << endl;
	}

	return 0;
}