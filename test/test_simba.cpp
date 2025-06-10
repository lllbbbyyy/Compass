#include <iostream>
#include <memory>
#include <iomanip>
#include "layer_engine.h"
#include"core.h"
#include"nns/nns.h"
#include"noc.h"
#include"debug.h"
#include "compass/request_generator.h"
#include "nns/nns.h"
#include "model_engine.h"
#include "compass/utils_compass.h"
#include "tqdm/tqdm.h"

using namespace std;


int main()
{
    std::cout << std::fixed << std::setprecision(2);
    
	int init_chip_number=36;
	int batch_size=64;
	int micro_batch_size=2;

	vector<shared_ptr<CoreMapper>> chips;
	for(int i=0;i<init_chip_number;i++){
		chips.emplace_back(createPolarCoreMapper(1024,1 MB));
	}
	DEBUG("chips created");

	auto noc=createNoC(6,6,128,64,4);
	DEBUG("noc created");

	ReqGenerator generator(batch_size);
	vector<shared_ptr<Network>> batched_models;
	auto batches = generator.generateReq(micro_batch_size);
	for(int i : tqdm(batch_size/micro_batch_size)){
    	//auto n=create_GPT3(batches[i],32,4096,32,128);
		auto n=create_GPT3(batches[i],1,256,8,32);
		batched_models.emplace_back(n);
	}
	size_t layer_lens=batched_models[0]->len();
	DEBUG("models created",layer_lens);

	vector<int> segmentation;
	vector<cidx_t> layerToChip;
	for(size_t i=0;i<layer_lens;i++){
		if(i!=layer_lens-1)
		{
			if((i+1)%init_chip_number==0)
				segmentation.push_back(1);
			else
				segmentation.push_back(0);
		}
		layerToChip.push_back(i%init_chip_number);
	}
	DEBUG("segmentation and layerToChip created");

	auto modelEngine=CompassModelEngine(batched_models,chips,noc,segmentation,layerToChip);

    auto [latency,energy]=modelEngine.calcLatencyAndEnergy();
    auto mc=modelEngine.calcMonetaryCost();

	cout<<"test1"<<endl;
	cout<<"latency: "<<latency<<" energy: "<<energy<<" mc: "<<mc<<endl;
	// cout<<"latency detail:"<<endl;
    // for(size_t i=0;i<modelEngine.latencyDetail.size();i++){
	// 	cout<<"	core "<<i<<": "<<endl;
	// 	for(auto& detail:modelEngine.latencyDetail[i]){
	// 		cout<<"	"<<detail;
	// 	}
	// }
	// cout<<"energy detail:"<<endl;
    // for(size_t i=0;i<modelEngine.energyDetail.size();i++){
	// 	cout<<"	core "<<i<<": "<<endl;
	// 	for(auto& detail:modelEngine.energyDetail[i]){
	// 		cout<<"		"<<detail;
	// 	}
	// }
	cout<<"mc detail:"<<endl;
	cout<<modelEngine.mcCost<<endl;

}