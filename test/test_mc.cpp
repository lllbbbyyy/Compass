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

    auto network=gen_convs(numCores);

    auto noc=createNoC(xlen,ylen,13,8,4);
    vector<shared_ptr<CoreMapper>> coreMappers;
    for(int i=0;i<numCores;i++){
        coreMappers.emplace_back(createPolarCoreMapper(1024,1 MB));
    }

    //create model engine
    auto modelEngine=CompassModelEngine({network},coreMappers,noc,{0,0,0},{0,1,2,3});
    auto latency=modelEngine.calcLatency();
    auto energy=modelEngine.calcEnergy();
    auto [l,e]=modelEngine.calcLatencyAndEnergy();
    assert(latency==l);
    assert(energy==e);
    auto mc=modelEngine.calcMonetaryCost();

    cout<<latency<<" "<<energy<<" "<<mc<<endl;
    cout<<modelEngine.mcCost<<endl;
}