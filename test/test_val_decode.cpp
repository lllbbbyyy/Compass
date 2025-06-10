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

using namespace std;


int main()
{
    std::cout << std::fixed << std::setprecision(2);
    
    //prepare for model
    // ReqGenerator generator(4);
    // auto batches = generator.generateReq(2);
    // vector<vector<Network>> networks;
    // for(auto& batch:batches){
    //     networks.emplace_back(create_GPT3(batch,1,256,8,32));
    // }
    mlen_t xlen=6,ylen=6;
    cidx_t numCores=xlen*ylen;

    auto network=create_LLM_block_decode(16, 256, 16, 1024);

    auto noc=createNoC(xlen,ylen,128,64,1);
    vector<shared_ptr<CoreMapper>> coreMappers;
    for(int i=0;i<numCores;i++){
        coreMappers.emplace_back(createPolarCoreMapper(1024,100 MB));
    }


    //create model engine
	std::vector<int> segmentation;
	std::vector<cidx_t> layerToChip;
	for(lid_t i=0;i<network->len();i++){
		if(i!=network->len()-1)
		{
			if((i+1)%numCores==0)
				segmentation.push_back(1);
			else
				segmentation.push_back(0);
		}
		layerToChip.push_back(i%numCores);
	}
    auto modelEngine=CompassModelEngine({network},coreMappers,noc,segmentation,layerToChip);
    auto [latency,energy]=modelEngine.calcLatencyAndEnergy();
    auto mc=modelEngine.calcMonetaryCost();

    cout<<latency<<" "<<energy<<" "<<mc<<endl;
    cout<<modelEngine.mcCost<<endl;
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
}