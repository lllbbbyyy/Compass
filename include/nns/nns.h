#ifndef NNS_H
#define NNS_H

#include "network.h"
#include "util.h"
#include "compass/request_generator.h"
#include <string>
#include <vector>

std::shared_ptr<Network> gen_single_conv();

std::shared_ptr<Network> gen_convs(int num);

std::shared_ptr<Network> GEMM4(int input_size);

std::shared_ptr<Network> create_motivation_two_layer(
    len_t layer_a_m,
    len_t layer_a_k,
    len_t layer_a_n,
    len_t layer_b_m,
    len_t layer_b_k,
    len_t layer_b_n,
    len_t layer_b_heads);

std::shared_ptr<Network> create_GPT3(const std::vector<Req>& reqs,len_t n_layers,len_t d_model,len_t n_heads,len_t d_head,len_t d_ffn,len_t d_model_tiling_size=4096,len_t d_ffn_tiling_size=4096,const std::string& mapping_merge_mode="none");

std::shared_ptr<Network> create_GPT3_merged(const std::vector<Req>& reqs,len_t n_layers,len_t d_model,len_t n_heads,len_t d_head,len_t d_ffn,const std::string& mapping_merge_mode="stage",len_t d_model_tiling_size=0,len_t d_ffn_tiling_size=0,const std::string& qkv_projection_mode="separate",len_t tensor_parallel=0);

std::shared_ptr<Network> create_llama3(const std::vector<Req>& reqs,len_t n_layers,len_t d_model,len_t n_heads,len_t d_head,len_t n_kv_heads,len_t d_ffn,len_t d_model_tiling_size=4096,len_t d_ffn_tiling_size=4096,const std::string& mapping_merge_mode="none");

std::shared_ptr<Network> create_llama3_merged(const std::vector<Req>& reqs,len_t n_layers,len_t d_model,len_t n_heads,len_t d_head,len_t n_kv_heads,len_t d_ffn,const std::string& mapping_merge_mode="stage",len_t d_model_tiling_size=0,len_t d_ffn_tiling_size=0,const std::string& qkv_projection_mode="separate",len_t tensor_parallel=0);

std::shared_ptr<Network> create_LLM_block_decode(len_t num_heads, len_t d_head, len_t kv_head_num, len_t seq_len);

std::shared_ptr<Network> create_LLM_block_prefill(len_t num_heads, len_t d_head, len_t group_num, len_t seq_len);

#endif // NNS_H
