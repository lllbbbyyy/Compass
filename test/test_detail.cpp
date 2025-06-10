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
#include "json.hpp"

using namespace std;


int main()
{
    std::cout << std::fixed << std::setprecision(2);
    
	int init_chip_number=36;
	int batch_size=64;
	int micro_batch_size=1;

	vector<shared_ptr<CoreMapper>> chips;
	for(int i=0;i<init_chip_number;i++){
		chips.emplace_back(createPolarCoreMapper(1024,100 MB));
	}
	DEBUG("chips created");

	auto noc=createNoC(6,6,256,128,4);
	DEBUG("noc created");

	ReqGenerator generator(batch_size);
	vector<shared_ptr<Network>> batched_models;
	auto batches = generator.generateReq(micro_batch_size);
	for(int i : tqdm(batch_size/micro_batch_size)){
    	DEBUG("batch",i);
		for(auto& req:batches[i]){
			DEBUG("	",req);
		}
	}
	// exit(0);
	for(int i : tqdm(batch_size/micro_batch_size)){
    	//auto n=create_GPT3(batches[i],32,4096,32,128);
		auto n=create_GPT3(batches[i],1,4096,32,128);
		//auto n =gen_convs(36);
		//auto n=create_GPT3(batches[i],1,256,8,32);
		batched_models.emplace_back(n);
	}
	size_t layer_lens=batched_models[0]->len();
    for(size_t i=0;i<layer_lens;i++){
        DEBUG("layer",i,batched_models[0]->getNode(i).name());
    }
	DEBUG("models created",layer_lens);


	// vector<int> segmentation;
	// vector<vector<cidx_t>> layerToChip;
	// for(size_t i=0;i<layer_lens;i++){
	// 	if(i!=layer_lens-1)
	// 	{
	// 		if((i+1)%init_chip_number==0)
	// 			segmentation.push_back(1);
	// 		else
	// 			segmentation.push_back(0);
	// 	}
	// }
	// for(size_t j=0;j<batched_models.size();j++){
	// 	layerToChip.push_back({});

	// 	for(size_t i=0;i<layer_lens;i++){
	// 		layerToChip[j].push_back(ThreadSafeRandom::rand_int(0,init_chip_number-1));
	// 		//layerToChip.push_back(i%init_chip_number);
	// 	}
	// }

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
	cout<<"latency detail:"<<endl;
	nlohmann::json j;
    for(size_t i=0;i<modelEngine.latencyDetail.size();i++){
		//cout<<"	core "<<i<<": "<<endl;
		j["core"+to_string(i)]=nlohmann::json::array();
		for(auto& detail:modelEngine.latencyDetail[i]){
			// cout<<"		"<<detail;
			nlohmann::json temp;
			temp["layerID"]=detail.layerID;
			temp["batchID"]=detail.batchID;
			temp["latencyBegin"]=detail.latencyBegin;
			temp["latencyEnd"]=detail.latencyEnd;
			j["core"+to_string(i)].push_back(temp);
		}
	}
	std::ofstream o("tmp/latency_detail.json");
	o << std::setw(4) << j << std::endl;

	cout<<"energy detail:"<<endl;
    // for(size_t i=0;i<modelEngine.energyDetail.size();i++){
	// 	cout<<"	core "<<i<<": "<<endl;
	// 	for(auto& detail:modelEngine.energyDetail[i]){
	// 		cout<<"		"<<detail;
	// 	}
	// }
	cout<<"mc detail:"<<endl;
	cout<<modelEngine.mcCost<<endl;

}