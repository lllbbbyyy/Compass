#include "nns/nns.h"
#include <string>
#include <cassert>
#include <memory>

std::shared_ptr<Network> gen_convs(int num){
    assert(num>0);
    auto n=std::make_shared<Network>();
    InputData input("input", fmap_shape(256,14));
    
    auto prev=n->add(NLAYER("conv1", Conv, C=256, K=256, H=14, R=3), {}, 0, {input});
    for(int i=1;i<num;i++){
        prev=n->add(NLAYER("conv"+std::to_string(i+1), Conv, C=256, K=256, H=14, R=3), {prev});
    }
    return n;
}

std::shared_ptr<Network> GEMM4(int input_size){
    auto n=std::make_shared<Network>();
    InputData input("input", fmap_shape(256,input_size));
    
    auto conv1=n->add(NLAYER("gemm1", Conv, C=256, K=256, H=input_size,W=1), {}, 0, {input});
    auto conv2=n->add(NLAYER("gemm2", Conv, C=256, K=256, H=input_size,W=1), {conv1});
    auto conv3=n->add(NLAYER("gemm3", Conv, C=256, K=128, H=input_size,W=1), {conv1});
    n->add(NLAYER("gemm4", Conv, C=input_size, K=128, H=256,W=1), {conv2},0,{},{conv3});
    return n;
}
