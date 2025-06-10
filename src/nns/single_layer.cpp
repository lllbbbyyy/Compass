#include "nns/nns.h"

#include<memory>

std::shared_ptr<Network> gen_single_conv(){
    auto n=std::make_shared<Network>();
    InputData input("input", fmap_shape(256,14));
    
    n->add(NLAYER("conv", Conv, C=256, K=256, H=14, R=3), {}, 0, {input});
    return n;
}
