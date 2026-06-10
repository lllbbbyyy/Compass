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

std::shared_ptr<Network> create_motivation_two_layer(
    len_t layer_a_m,
    len_t layer_a_k,
    len_t layer_a_n,
    len_t layer_b_m,
    len_t layer_b_k,
    len_t layer_b_n,
    len_t layer_b_heads)
{
    assert(layer_b_heads > 0);
    assert(layer_a_n % 3 == 0);
    auto n = std::make_shared<Network>();
    InputData input_a("input_a", fmap_shape(layer_a_k, layer_a_m, 1));

    auto qkv_mapping = n->createMappingNode("layerA_qkv_gen");
    auto attn_mapping = n->createMappingNode("layerB_attention_score");

    const len_t projection_n = layer_a_n / 3;
    auto k = n->add(qkv_mapping, NLAYER("layerA_k_gen", Conv, C=layer_a_k, K=projection_n, H=layer_a_m, W=1), {}, 0, {input_a});
    n->add(qkv_mapping, NLAYER("layerA_v_gen", Conv, C=layer_a_k, K=projection_n, H=layer_a_m, W=1), {}, 0, {input_a});
    auto q = n->add(qkv_mapping, NLAYER("layerA_q_gen", Conv, C=layer_a_k, K=projection_n, H=layer_a_m, W=1), {}, 0, {input_a});
    for(len_t head = 0; head < layer_b_heads; ++head){
        n->add(attn_mapping, NLAYER("layerB_attention_score_h" + std::to_string(head), Conv, C=layer_b_k, K=layer_b_n, H=layer_b_m, W=1), {q}, 0, {}, {k});
    }
    return n;
}
