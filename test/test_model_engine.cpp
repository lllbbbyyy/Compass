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

using namespace std;

shared_ptr<EyerissMapper> createEyerissCoreMapper(){
    Core::numMac_t LR_mac_num = 64;
    energy_t LR_mac_cost = 0.0873; //IEEE FP16
    EyerissCore::PESetting s2(32, 32, 0.018);
    EyerissCore::Bus ibus(0.018, 64);
    EyerissCore::Bus wbus(0.018, 64);
    EyerissCore::Bus pbus(0.018, 64); // ifmap RC, weight RCK, psum RK
    EyerissCore::Buses eBus{ibus, wbus, pbus};

    EyerissCore::Buffers eBuf;

    eBuf.al1.Size = 32;
    eBuf.pl1.Size = 1;
    eBuf.wl1.Size = 128;
    eBuf.ul2.Size = 1024 KB;

    eBuf.al1.RCost = 0.0509 * 8; //8bit IO single port
    eBuf.al1.WCost = 0.0506 * 8;//0.045;
    eBuf.wl1.RCost = 0.0545 * 8; //Using 2 banks of 64
    eBuf.wl1.WCost = 0.054 * 8;//0.090;
    eBuf.pl1.RCost = eBuf.pl1.WCost = 0.0;
    eBuf.ul2.RCost = 0.1317125 * 8;
    eBuf.ul2.WCost = 0.234025 * 8;

    auto core = make_shared<EyerissCore>(s2, LR_mac_num, LR_mac_cost, eBus, eBuf);
    return make_shared<EyerissMapper>(core);
}

shared_ptr<NoC> createNoC(mlen_t xlen,mlen_t ylen,int dram_num=2){
    double hop_cost=0.7 * 8;
    double DRAM_acc_cost=7.5 * 8;
    bw_t DRAM_bw_each=1;
    bw_t NoC_bw=24;

    std::vector<bw_t> dram_bws(dram_num,DRAM_bw_each);
    std::vector<pos_t> dram_router_list;
    std::vector<std::vector<pos_t>> dram_list;
    // total 2*ylen dram router，every side has dram_num/2 dram，every dram has ylen/(dram_num/2) dram router
    // Sets DRAM router
    size_t router_num_per_dram=ylen/(dram_num/2);
    std::vector<pos_t> routers;
	for(mlen_t y=0; y<ylen; ++y){
        dram_router_list.push_back({0, y});
        if(y%router_num_per_dram==0&&!routers.empty())
        {
            dram_list.push_back(routers);
            routers.clear();
        }
        routers.push_back({0, y});
	}
    if(!routers.empty())
    {
        dram_list.push_back(routers);
        routers.clear();
    }
	for(mlen_t y=0; y<ylen; ++y){
        dram_router_list.push_back({static_cast<mlen_t>(xlen-1), y});
        if(y%router_num_per_dram==0&&!routers.empty())
        {
            dram_list.push_back(routers);
            routers.clear();
        }
        routers.push_back({static_cast<mlen_t>(xlen-1), y});
	}
    if(!routers.empty())
    {
        dram_list.push_back(routers);
        routers.clear();
    }
    DEBUG(dram_list.size());
    DEBUG(dram_router_list.size());
    DEBUG(dram_list[0].size());
    DEBUG(dram_list[1].size());
    DEBUG(dram_list[0][0]);
    DEBUG(dram_list[1][0]);
    return make_shared<NoC>(xlen,ylen,hop_cost,DRAM_acc_cost,NoC_bw,dram_bws,dram_router_list,dram_list);
}
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
    mlen_t xlen=2,ylen=2;
    cidx_t numCores=xlen*ylen;

    auto network1=gen_convs(numCores);
	auto network2=gen_convs(numCores);

    auto noc=createNoC(xlen,ylen);
    vector<shared_ptr<CoreMapper>> coreMappers;
    for(int i=0;i<numCores;i++){
        coreMappers.emplace_back(createEyerissCoreMapper());
    }
    
    //create model engine
    auto modelEngine=CompassModelEngine({network1,network2},coreMappers,noc,{0,0,0},{0,1,1,3});
    
    auto latency=modelEngine.calcLatency();
    auto energy=modelEngine.calcEnergy();
    auto [l,e]=modelEngine.calcLatencyAndEnergy();
    assert(latency==l);
    assert(energy==e);
    auto mc=modelEngine.calcMonetaryCost();

	cout<<"test1"<<endl;
	cout<<"latency: "<<latency<<" energy: "<<energy<<" mc: "<<mc<<endl;
	cout<<"latency detail:"<<endl;
    for(size_t i=0;i<modelEngine.latencyDetail.size();i++){
		cout<<"	core "<<i<<": "<<endl;
		for(auto& detail:modelEngine.latencyDetail[i]){
			cout<<"	"<<detail;
		}
	}
	cout<<"energy detail:"<<endl;
    for(size_t i=0;i<modelEngine.energyDetail.size();i++){
		cout<<"	core "<<i<<": "<<endl;
		for(auto& detail:modelEngine.energyDetail[i]){
			cout<<"		"<<detail;
		}
	}
	cout<<"mc detail:"<<endl;
	cout<<modelEngine.mcCost<<endl;

	modelEngine.setSegmentation({0,0,0},std::vector<std::vector<cidx_t>>(2,{0,1,2,3}));
	//modelEngine.setLayerToChip({0,1,2,3});
	latency=modelEngine.calcLatency();
    energy=modelEngine.calcEnergy();
    mc=modelEngine.calcMonetaryCost();
	cout<<"test2"<<endl;
	cout<<"latency: "<<latency<<" energy: "<<energy<<" mc: "<<mc<<endl;
	cout<<"latency detail:"<<endl;
    for(size_t i=0;i<modelEngine.latencyDetail.size();i++){
		cout<<"	core "<<i<<": "<<endl;
		for(auto& detail:modelEngine.latencyDetail[i]){
			cout<<"	"<<detail;
		}
	}
	cout<<"energy detail:"<<endl;
    for(size_t i=0;i<modelEngine.energyDetail.size();i++){
		cout<<"	core "<<i<<": "<<endl;
		for(auto& detail:modelEngine.energyDetail[i]){
			cout<<"		"<<detail;
		}
	}
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
	cout<<"mc detail:"<<endl;
	cout<<modelEngine.mcCost<<endl;
}