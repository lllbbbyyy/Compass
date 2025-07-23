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

	CompassModelEngine model_engine(batched_models, chips, noc,{1,0,0},{{0,0,2,2},{1,0,1,2},{2,0,1,2},{3,1,3,2}});
	model_engine.calcLatencyAndEnergy();
	auto j=model_engine.get_latency_detail();
	std::ofstream o("tmp/GEMM4_latency_detail.json");
    o << std::setw(4) << j << std::endl;

	return 0;
}