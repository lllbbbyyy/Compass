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

void normal_run(){
	int init_chip_number = 4;
	vector<shared_ptr<CoreMapper>> chips;
	for (int i = 0; i < init_chip_number; i++)
	{
		chips.emplace_back(createPolarCoreMapper(256, 100 MB));
		// chips.emplace_back(createEyerissCoreMapper(1024,100 MB));
	}

	auto noc = createNoC(2, 2, 256, 128, 2);
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
	auto [l,e]=model_engine.calcLatencyAndEnergy();
	cout<<l*e/1e18<<endl;
	auto j=model_engine.get_latency_detail();
	std::ofstream o("tmp/GEMM4_latency_detail.json");
    o << std::setw(4) << j << std::endl;
}

void dp(){
	int init_chip_number = 4;
	int C=init_chip_number;
	vector<shared_ptr<CoreMapper>> chips;
	for (int i = 0; i < init_chip_number; i++)
	{
		chips.emplace_back(createPolarCoreMapper(256, 100 MB));
		// chips.emplace_back(createEyerissCoreMapper(1024,100 MB));
	}

	auto noc = createNoC(2, 2, 256, 128, 2);
	vector<shared_ptr<Network>> batched_models;

	vector v{64,64,128,128,128,128,128,256};
	for(auto vv:v){
		auto n = GEMM4(vv);
		batched_models.emplace_back(n);
	}

	size_t L = batched_models[0]->len();
	size_t B = 8;
	vector<int> segmentation(L-1, 0);
	vector<vector<int>> layer_to_chip(B, vector<int>(L, -1));
	for(int i=0;i<=B-1;i++){
		for(int j=0;j<=L-1;j++){
			layer_to_chip[i][j] = i%C;
		}
	}
	cout<<"segemetation:"<<endl;
	for(auto s:segmentation){
		cout<<s<<" ";
	}
	cout<<endl;
	cout<<"layer2chip:"<<endl;
	for(auto &l:layer_to_chip){
		for(auto c:l){
			cout<<c<<" ";
		}
		cout<<endl;
	}

	CompassModelEngine model_engine(batched_models, chips, noc,segmentation, layer_to_chip);
	auto [l,e]=model_engine.calcLatencyAndEnergy();
	cout<<l*e/1e18<<endl;
	auto j=model_engine.get_latency_detail();
	std::ofstream o("tmp/GEMM4_latency_detail_dp.json");
    o << std::setw(4) << j << std::endl;
}

void pp(){
	int init_chip_number = 4;
	int C=init_chip_number;
	vector<shared_ptr<CoreMapper>> chips;
	for (int i = 0; i < init_chip_number; i++)
	{
		chips.emplace_back(createPolarCoreMapper(256, 100 MB));
		// chips.emplace_back(createEyerissCoreMapper(1024,100 MB));
	}

	auto noc = createNoC(2, 2, 256, 128, 2);
	vector<shared_ptr<Network>> batched_models;

	auto n1 = GEMM4(128);
	batched_models.emplace_back(n1);
	auto n2 = GEMM4(256);
	batched_models.emplace_back(n2);
	auto n3 = GEMM4(256);
	batched_models.emplace_back(n3);
	auto n4 = GEMM4(384);
	batched_models.emplace_back(n4);

	size_t L = batched_models[0]->len();
	size_t B = 8;
	vector<int> segmentation(L-1, 0);
	vector<vector<int>> layer_to_chip(B/2, vector<int>(L, -1));
	for(int i=0;i<=L-2;i++){
		if((i+1)%C==0){
			segmentation[i] = 1; // segmentation point
		}

	}
	for(int j=0;j<=L-1;j++){
		for(int i=0;i<=B/2-1;i++){
			layer_to_chip[i][j]=j%C;
		}
	}
	cout<<"segemetation:"<<endl;
	for(auto s:segmentation){
		cout<<s<<" ";
	}
	cout<<endl;
	cout<<"layer2chip:"<<endl;
	for(auto &l:layer_to_chip){
		for(auto c:l){
			cout<<c<<" ";
		}
		cout<<endl;
	}
	CompassModelEngine model_engine(batched_models, chips, noc,segmentation, layer_to_chip);
	auto [l,e]=model_engine.calcLatencyAndEnergy();
	cout<<l*e/1e18<<endl;
	auto j=model_engine.get_latency_detail();
	std::ofstream o("tmp/GEMM4_latency_detail_pp.json");
    o << std::setw(4) << j << std::endl;
}

void mp(){
	int init_chip_number = 4;
	int C=init_chip_number;
	vector<shared_ptr<CoreMapper>> chips;
	for (int i = 0; i < init_chip_number; i++)
	{
		chips.emplace_back(createPolarCoreMapper(256, 100 MB));
		// chips.emplace_back(createEyerissCoreMapper(1024,100 MB));
	}

	auto noc = createNoC(2, 2, 256, 128, 2);
	vector<shared_ptr<Network>> batched_models;

	vector v{64,64,128,128,128,128,128,256};
	int total_v=0;
	for(auto vv:v){
		total_v+=vv;
	}
	auto n = GEMM4(total_v);
	batched_models.emplace_back(n);

	size_t L = batched_models[0]->len();
	size_t B = batched_models.size();
	vector<int> segmentation(L-1, 0);
	vector<vector<int>> layer_to_chip(1, vector<int>(L, -1));
	for(int i=0;i<=L-1;i++){
		layer_to_chip[0][i] = i%C;
	}
	cout<<"segemetation:"<<endl;
	for(auto s:segmentation){
		cout<<s<<" ";
	}
	cout<<endl;
	cout<<"layer2chip:"<<endl;
	for(auto &l:layer_to_chip){
		for(auto c:l){
			cout<<c<<" ";
		}
		cout<<endl;
	}
	CompassModelEngine model_engine(batched_models, chips, noc,segmentation, layer_to_chip);
	auto [l,e]=model_engine.calcLatencyAndEnergy();
	cout<<l*e/1e18<<endl;
	auto j=model_engine.get_latency_detail();
	std::ofstream o("tmp/GEMM4_latency_detail_mp.json");
    o << std::setw(4) << j << std::endl;
}

int main()
{

	normal_run();
	// DP
	DEBUG("DP");
	dp();
	// MP
	DEBUG("MP");
	mp();
	// PP
	DEBUG("PP");
	pp();
	return 0;
}