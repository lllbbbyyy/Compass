#include "network.h"

#include <cassert>

#include "util.h"
#include "compass/request_generator.h"
#include "debug.h"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

namespace {
constexpr Network::mapping_id_t NO_MAPPING_NODE = -1;

std::string normalize_mapping_merge_mode(const std::string& mapping_merge_mode)
{
	if(mapping_merge_mode.empty() || mapping_merge_mode == "none"){
		return "none";
	}
	if(mapping_merge_mode == "stage"){
		return "stage";
	}
	if(mapping_merge_mode == "stage_post" || mapping_merge_mode == "stage-post"){
		return "stage_post";
	}
	throw std::logic_error("Unsupported mapping merge mode: " + mapping_merge_mode);
}

bool merge_stages(const std::string& mapping_merge_mode)
{
	return mapping_merge_mode != "none";
}

bool merge_post_processing(const std::string& mapping_merge_mode)
{
	return mapping_merge_mode == "stage_post";
}

Network::mapping_id_t create_mapping_if_needed(
	const std::shared_ptr<Network>& n,
	const std::string& mapping_merge_mode,
	const std::string& name
)
{
	return merge_stages(mapping_merge_mode) ? n->createMappingNode(name) : NO_MAPPING_NODE;
}

lid_t add_to_mapping(
	const std::shared_ptr<Network>& n,
	Network::mapping_id_t mapping_id,
	Layer* l,
	const Network::layer_set& ifmPrevs={},
	bwidth_t width=0,
	const std::vector<InputData>& ifm_input_data={},
	const Network::layer_set& wgtPrevs={},
	const std::vector<InputData>& wgtInputData={}
)
{
	if(mapping_id >= 0){
		return n->add(mapping_id, l, ifmPrevs, width, ifm_input_data, wgtPrevs, wgtInputData);
	}
	return n->add(l, ifmPrevs, width, ifm_input_data, wgtPrevs, wgtInputData);
}
}

std::shared_ptr<Network> create_GPT3(const std::vector<Req>& reqs,len_t n_layers,len_t d_model,len_t n_heads,len_t d_head,len_t d_ff,len_t d_model_tiling_size,len_t d_ff_tiling_size,const std::string& mapping_merge_mode)
{
	const std::string merge_mode = normalize_mapping_merge_mode(mapping_merge_mode);
	auto n=std::make_shared<Network>();
	assert(d_model==n_heads*d_head);
	len_t seq_lens_sum = 0;
	for(auto& req : reqs){
		seq_lens_sum += req.lens;
	}

	// Input layer for the new token
	InputData input("input", fmap_shape(d_model, seq_lens_sum, 1));
	InputData pos_encoding("pos_encoding", fmap_shape(d_model, seq_lens_sum, 1));

	lid_t prev_layer=n->add(NLAYER("input_add", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), {},0,{input,pos_encoding});
	
	for(len_t i=0;i<n_layers;i++){
		std::string layer_name="layer"+std::to_string(i);
		auto qkv_gen_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_QKV_Gen");
		auto attn_core_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_Attn_Core");
		Network::layer_set QKVs;
		for(len_t j=0;j<n_heads;j++)
		{
			// Network::lid_t Q, K, V, newK, newKt, Kt, newV, newVt, Vt,  QK, QK_elt, QKV;
			std::string name = layer_name+"_head"+std::to_string(j);

			lid_t Q, K, V,  QK, QK_elt, QKV;

			//DEBUG("seq_lens_sum",seq_lens_sum);
			Q = add_to_mapping(n, qkv_gen_mapping, NLAYER(name+"_Q", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {prev_layer});
			K = add_to_mapping(n, qkv_gen_mapping, NLAYER(name+"_K", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {prev_layer});
			n->getNode(K).mustWriteDRAM=true;
			V = add_to_mapping(n, qkv_gen_mapping, NLAYER(name+"_V", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {prev_layer});
			n->getNode(V).mustWriteDRAM=true;

			//split by req
			for(size_t k=0;k<reqs.size();k++)
			{
				const auto& req=reqs[k];
				std::string name_mul=name+"_req"+std::to_string(k);
				if(req.type==Req::Type::Prefill||(req.type==Req::Type::ChunkedPrefill&&req.his_lens==0))
				{
					QK = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QK_P", Conv, C=d_head,K=req.lens, H=req.lens, W=1), {Q},0,{}, {K});
					QK_elt = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QK_elt_P", PTP, K=req.lens,H=req.lens, W=1), {QK});
					QKV = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QKV_P", Conv, C=req.lens,K=d_head, H=req.lens, W=1), {QK_elt},0,{}, {V});
				}
				else if(req.type==Req::Type::Decode||(req.type==Req::Type::ChunkedPrefill&&req.his_lens!=0))
				{
					// all K and V from DRAM after calculating newK and newV
					InputData k_cache("k_cache", fmap_shape(req.his_lens, d_head, 1)); // transposed K cache
					InputData v_cache("v_cache", fmap_shape(req.his_lens, d_head, 1)); // V cache
					QK = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QK_D", Conv, C=d_head,K=req.his_lens+req.lens, H=req.lens, W=1), {Q},0,{}, {K}, {k_cache});
					QK_elt = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QK_elt_D", PTP, K=req.his_lens+req.lens,H=req.lens, W=1), {QK});
					QKV = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QKV_D", Conv, C=req.his_lens+req.lens,K=d_head, H=req.lens, W=1), {QK_elt},0,{}, {V}, {v_cache});
				}
				QKVs.push_back(QKV);
			}

		}
		
		// lid_t attn_output = n->add(NLAYER(layer_name + "_out_proj", Conv, C=d_model, H=seq_lens_sum, W=1), QKVs);
		assert(d_ff % d_ff_tiling_size == 0); // Ensure d_ff is divisible by ffn_tiling_size
		assert(d_model % d_model_tiling_size == 0); // Ensure d_model is divisible by ffn_tiling_size

		Network::layer_set attn_output;
		auto out_proj_mapping = create_mapping_if_needed(
			n,
			merge_mode,
			merge_post_processing(merge_mode) ? layer_name+"_Out_Proj_Post" : layer_name+"_Out_Proj"
		);
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_out_proj_tiling_"+std::to_string(j);
			attn_output.push_back(add_to_mapping(n, out_proj_mapping, NLAYER(name, Conv, C=d_model, K=d_model_tiling_size, H=seq_lens_sum, W=1), QKVs));
		}
		attn_output.push_back(prev_layer);
		lid_t res1;
		if(merge_post_processing(merge_mode)){
			res1 = add_to_mapping(n, out_proj_mapping, NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), attn_output);
		}
		else{
			res1 = n->add(NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), attn_output);
		}

		// // Residual connection after attention
		// lid_t res1 = n->add(NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), {prev_layer, attn_output});
		
		// LayerNorm after first residual
		lid_t norm1;
		if(merge_post_processing(merge_mode)){
			norm1 = add_to_mapping(n, out_proj_mapping, NLAYER(layer_name+"_norm1", PTP, K=d_model, H=seq_lens_sum, W=1), {res1});
		}
		else{
			norm1 = n->add(NLAYER(layer_name+"_norm1", PTP, K=d_model, H=seq_lens_sum, W=1), {res1});
		}
		// Feed-Forward Network
	
		Network::layer_set ff1,ff2;
		auto ffn1_act_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_FFN1_Act");
		for(len_t j=0;j<d_ff/d_ff_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn1_tiling"+std::to_string(j);
			ff1.push_back(add_to_mapping(n, ffn1_act_mapping, NLAYER(name, Conv, C=d_model, K=d_ff_tiling_size, H=seq_lens_sum, W=1), {norm1}));
		}
		lid_t gelu = add_to_mapping(n, ffn1_act_mapping, NLAYER(layer_name+"_GeLU", PTP, K=d_ff, H=seq_lens_sum, W=1), ff1);
		auto ffn2_mapping = create_mapping_if_needed(
			n,
			merge_mode,
			merge_post_processing(merge_mode) ? layer_name+"_FFN2_Post" : layer_name+"_FFN2"
		);
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn2_tiling"+std::to_string(j);
			ff2.push_back(add_to_mapping(n, ffn2_mapping, NLAYER(name, Conv, C=d_ff, K=d_model_tiling_size, H=seq_lens_sum, W=1), {gelu}));
		}
		ff2.push_back(norm1);
		lid_t res2;
		if(merge_post_processing(merge_mode)){
			res2 = add_to_mapping(n, ffn2_mapping, NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), ff2);
		}
		else{
			res2 = n->add(NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), ff2);
		}
		lid_t norm2;
		if(merge_post_processing(merge_mode)){
			norm2 = add_to_mapping(n, ffn2_mapping, NLAYER(layer_name+"_norm2", PTP, K=d_model, H=seq_lens_sum, W=1), {res2});
		}
		else{
			norm2 = n->add(NLAYER(layer_name+"_norm2", PTP, K=d_model, H=seq_lens_sum, W=1), {res2});
		}
		// lid_t ff1 = n->add(NLAYER(layer_name+"_ffn1", Conv, C=d_model, K=d_ff, H=seq_lens_sum, W=1), {norm1});
		// lid_t gelu = n->add(NLAYER(layer_name+"_GeLU", PTP, K=d_ff, H=seq_lens_sum, W=1), {ff1});
		// lid_t ff2 = n->add(NLAYER(layer_name+"_ffn2", Conv, C=d_ff, K=d_model, H=seq_lens_sum, W=1), {gelu});

		// // Residual connection after FFN
		// lid_t res2 = n->add(NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), {norm1, ff2});
		prev_layer = norm2;
	}

	return n;
};

std::shared_ptr<Network> create_GPT3_merged(const std::vector<Req>& reqs,len_t n_layers,len_t d_model,len_t n_heads,len_t d_head,len_t d_ff,const std::string& mapping_merge_mode,len_t d_model_tiling_size,len_t d_ff_tiling_size)
{
	const std::string merge_mode = normalize_mapping_merge_mode(mapping_merge_mode);
	auto n=std::make_shared<Network>();
	assert(d_model==n_heads*d_head);
	if(d_model_tiling_size == 0){
		d_model_tiling_size = d_model;
	}
	if(d_ff_tiling_size == 0){
		d_ff_tiling_size = d_ff;
	}
	assert(d_model % d_model_tiling_size == 0);
	assert(d_ff % d_ff_tiling_size == 0);
	len_t seq_lens_sum = 0;
	for(auto& req : reqs){
		seq_lens_sum += req.lens;
	}

	InputData input("input", fmap_shape(d_model, seq_lens_sum, 1));
	InputData pos_encoding("pos_encoding", fmap_shape(d_model, seq_lens_sum, 1));

	lid_t prev_layer=n->add(NLAYER("input_add", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), {},0,{input,pos_encoding});

	for(len_t i=0;i<n_layers;i++){
		std::string layer_name="layer"+std::to_string(i);
		auto qkv_gen_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_QKV_Gen");

		lid_t K = add_to_mapping(n, qkv_gen_mapping, NLAYER(layer_name+"_k_gen_merged", Conv, C=d_model, K=d_model, H=seq_lens_sum, W=1), {prev_layer});
		n->getNode(K).mustWriteDRAM=true;
		lid_t V = add_to_mapping(n, qkv_gen_mapping, NLAYER(layer_name+"_v_gen_merged", Conv, C=d_model, K=d_model, H=seq_lens_sum, W=1), {prev_layer});
		n->getNode(V).mustWriteDRAM=true;
		lid_t Q = add_to_mapping(n, qkv_gen_mapping, NLAYER(layer_name+"_q_gen_merged", Conv, C=d_model, K=d_model, H=seq_lens_sum, W=1), {prev_layer});

		auto attn_qk_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_Attn_QK");
		Network::layer_set QKElts;
		for(len_t j=0;j<n_heads;j++)
		{
			std::string name = layer_name+"_head"+std::to_string(j);

			for(size_t k=0;k<reqs.size();k++)
			{
				const auto& req=reqs[k];
				std::string name_mul=name+"_req"+std::to_string(k);
				lid_t QK, QK_elt;
				if(req.type==Req::Type::Prefill||(req.type==Req::Type::ChunkedPrefill&&req.his_lens==0))
				{
					QK = add_to_mapping(n, attn_qk_mapping, NLAYER(name_mul+"_QK_P", Conv, C=d_head,K=req.lens, H=req.lens, W=1), {Q},0,{}, {K});
					QK_elt = add_to_mapping(n, attn_qk_mapping, NLAYER(name_mul+"_QK_elt_P", PTP, K=req.lens,H=req.lens, W=1), {QK});
				}
				else
				{
					InputData k_cache("k_cache", fmap_shape(req.his_lens, d_head, 1));
					QK = add_to_mapping(n, attn_qk_mapping, NLAYER(name_mul+"_QK_D", Conv, C=d_head,K=req.his_lens+req.lens, H=req.lens, W=1), {Q},0,{}, {K}, {k_cache});
					QK_elt = add_to_mapping(n, attn_qk_mapping, NLAYER(name_mul+"_QK_elt_D", PTP, K=req.his_lens+req.lens,H=req.lens, W=1), {QK});
				}
				QKElts.push_back(QK_elt);
			}
		}

		auto attn_av_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_Attn_AV");
		Network::layer_set QKVs;
		size_t qk_elt_idx = 0;
		for(len_t j=0;j<n_heads;j++)
		{
			std::string name = layer_name+"_head"+std::to_string(j);
			lid_t QKV;

			for(size_t k=0;k<reqs.size();k++)
			{
				const auto& req=reqs[k];
				std::string name_mul=name+"_req"+std::to_string(k);
				const lid_t QK_elt = QKElts[qk_elt_idx++];
				if(req.type==Req::Type::Prefill||(req.type==Req::Type::ChunkedPrefill&&req.his_lens==0))
				{
					QKV = add_to_mapping(n, attn_av_mapping, NLAYER(name_mul+"_QKV_P", Conv, C=req.lens,K=d_head, H=req.lens, W=1), {QK_elt},0,{}, {V});
				}
				else if(req.type==Req::Type::Decode||(req.type==Req::Type::ChunkedPrefill&&req.his_lens!=0))
				{
					InputData v_cache("v_cache", fmap_shape(req.his_lens, d_head, 1));
					QKV = add_to_mapping(n, attn_av_mapping, NLAYER(name_mul+"_QKV_D", Conv, C=req.his_lens+req.lens,K=d_head, H=req.lens, W=1), {QK_elt},0,{}, {V}, {v_cache});
				}
				QKVs.push_back(QKV);
			}
		}

		auto out_proj_mapping = create_mapping_if_needed(
			n,
			merge_mode,
			merge_post_processing(merge_mode) ? layer_name+"_Out_Proj_Post" : layer_name+"_Out_Proj"
		);
		Network::layer_set attn_output;
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_out_proj_tiling_"+std::to_string(j);
			attn_output.push_back(add_to_mapping(n, out_proj_mapping, NLAYER(name, Conv, C=d_model, K=d_model_tiling_size, H=seq_lens_sum, W=1), QKVs));
		}
		lid_t res1;
		if(merge_post_processing(merge_mode)){
			attn_output.push_back(prev_layer);
			res1 = add_to_mapping(n, out_proj_mapping, NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), attn_output);
		}
		else{
			attn_output.push_back(prev_layer);
			res1 = n->add(NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), attn_output);
		}
		lid_t norm1;
		if(merge_post_processing(merge_mode)){
			norm1 = add_to_mapping(n, out_proj_mapping, NLAYER(layer_name+"_norm1", PTP, K=d_model, H=seq_lens_sum, W=1), {res1});
		}
		else{
			norm1 = n->add(NLAYER(layer_name+"_norm1", PTP, K=d_model, H=seq_lens_sum, W=1), {res1});
		}

		auto ffn1_act_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_FFN1_Act");
		Network::layer_set ff1,ff2;
		for(len_t j=0;j<d_ff/d_ff_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn1_tiling"+std::to_string(j);
			ff1.push_back(add_to_mapping(n, ffn1_act_mapping, NLAYER(name, Conv, C=d_model, K=d_ff_tiling_size, H=seq_lens_sum, W=1), {norm1}));
		}
		lid_t gelu = add_to_mapping(n, ffn1_act_mapping, NLAYER(layer_name+"_GeLU", PTP, K=d_ff, H=seq_lens_sum, W=1), ff1);
		auto ffn2_mapping = create_mapping_if_needed(
			n,
			merge_mode,
			merge_post_processing(merge_mode) ? layer_name+"_FFN2_Post" : layer_name+"_FFN2"
		);
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn2_tiling"+std::to_string(j);
			ff2.push_back(add_to_mapping(n, ffn2_mapping, NLAYER(name, Conv, C=d_ff, K=d_model_tiling_size, H=seq_lens_sum, W=1), {gelu}));
		}
		lid_t res2;
		if(merge_post_processing(merge_mode)){
			ff2.push_back(norm1);
			res2 = add_to_mapping(n, ffn2_mapping, NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), ff2);
		}
		else{
			ff2.push_back(norm1);
			res2 = n->add(NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), ff2);
		}
		lid_t norm2;
		if(merge_post_processing(merge_mode)){
			norm2 = add_to_mapping(n, ffn2_mapping, NLAYER(layer_name+"_norm2", PTP, K=d_model, H=seq_lens_sum, W=1), {res2});
		}
		else{
			norm2 = n->add(NLAYER(layer_name+"_norm2", PTP, K=d_model, H=seq_lens_sum, W=1), {res2});
		}
		prev_layer = norm2;
	}

	return n;
};

std::shared_ptr<Network> create_llama3(
    const std::vector<Req>& reqs,
    len_t n_layers,
    len_t d_model,
    len_t n_heads,
    len_t d_head,
    len_t n_kv_heads,  // GQA support
    len_t d_ff,        
    len_t d_model_tiling_size,
    len_t d_ff_tiling_size,
    const std::string& mapping_merge_mode
) {
	const std::string merge_mode = normalize_mapping_merge_mode(mapping_merge_mode);
	auto n=std::make_shared<Network>();
	assert(d_model==n_heads*d_head);
	assert(n_kv_heads <= n_heads && n_heads%n_kv_heads==0); // Ensure kv_heads is not greater than total heads
	len_t seq_lens_sum = 0;
	for(auto& req : reqs){
		seq_lens_sum += req.lens;
	}

	// Input layer for the new token
	InputData input("input", fmap_shape(d_model, seq_lens_sum, 1));
	InputData pos_encoding("pos_encoding", fmap_shape(d_model, seq_lens_sum, 1));

	lid_t prev_layer=n->add(NLAYER("input_add", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), {},0,{input,pos_encoding});
	
	for(len_t i=0;i<n_layers;i++){
		std::string layer_name="layer"+std::to_string(i);

		lid_t norm1 = n->add(NLAYER(layer_name+"_norm1", PTP, K=d_model, H=seq_lens_sum, W=1), {prev_layer});

		auto qkv_gen_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_QKV_Gen");
		auto attn_core_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_Attn_Core");
		Network::layer_set QKVs;
		lid_t Q, K=0, V=0,  QK, QK_elt, QKV;
		for(len_t j=0;j<n_heads;j++)
		{
			// Network::lid_t Q, K, V, newK, newKt, Kt, newV, newVt, Vt,  QK, QK_elt, QKV;
			std::string name = layer_name+"_head"+std::to_string(j);

			//DEBUG("seq_lens_sum",seq_lens_sum);
			Q = add_to_mapping(n, qkv_gen_mapping, NLAYER(name+"_Q", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {norm1});
			if(j%(n_heads/n_kv_heads)==0) // Only the first head in each kv group computes K and V
			{
				K = add_to_mapping(n, qkv_gen_mapping, NLAYER(name+"_K", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {norm1});
				n->getNode(K).mustWriteDRAM=true;
				V = add_to_mapping(n, qkv_gen_mapping, NLAYER(name+"_V", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {norm1});
				n->getNode(V).mustWriteDRAM=true;
			}
			//split by req
			for(size_t k=0;k<reqs.size();k++)
			{
				const auto& req=reqs[k];
				std::string name_mul=name+"_req"+std::to_string(k);
				if(req.type==Req::Type::Prefill||(req.type==Req::Type::ChunkedPrefill&&req.his_lens==0))
				{
					QK = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QK_P", Conv, C=d_head,K=req.lens, H=req.lens, W=1), {Q},0,{}, {K});
					QK_elt = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QK_elt_P", PTP, K=req.lens,H=req.lens, W=1), {QK});
					QKV = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QKV_P", Conv, C=req.lens,K=d_head, H=req.lens, W=1), {QK_elt},0,{}, {V});
				}
				else if(req.type==Req::Type::Decode||(req.type==Req::Type::ChunkedPrefill&&req.his_lens!=0))
				{
					// all K and V from DRAM after calculating newK and newV
					InputData k_cache("k_cache", fmap_shape(req.his_lens, d_head, 1)); // transposed K cache
					InputData v_cache("v_cache", fmap_shape(req.his_lens, d_head, 1)); // V cache
					QK = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QK_D", Conv, C=d_head,K=req.his_lens+req.lens, H=req.lens, W=1), {Q},0,{}, {K}, {k_cache});
					QK_elt = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QK_elt_D", PTP, K=req.his_lens+req.lens,H=req.lens, W=1), {QK});
					QKV = add_to_mapping(n, attn_core_mapping, NLAYER(name_mul+"_QKV_D", Conv, C=req.his_lens+req.lens,K=d_head, H=req.lens, W=1), {QK_elt},0,{}, {V}, {v_cache});
				}
				QKVs.push_back(QKV);
			}

		}
		
		// lid_t attn_output = n->add(NLAYER(layer_name + "_out_proj", Conv, C=d_model, H=seq_lens_sum, W=1), QKVs);
		assert(d_ff % d_ff_tiling_size == 0); // Ensure d_ff is divisible by ffn_tiling_size
		assert(d_model % d_model_tiling_size == 0); // Ensure d_model is divisible by ffn_tiling_size

		Network::layer_set attn_output;
		auto out_proj_mapping = create_mapping_if_needed(
			n,
			merge_mode,
			merge_post_processing(merge_mode) ? layer_name+"_Out_Proj_Post" : layer_name+"_Out_Proj"
		);
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_out_proj_tiling_"+std::to_string(j);
			attn_output.push_back(add_to_mapping(n, out_proj_mapping, NLAYER(name, Conv, C=d_model, K=d_model_tiling_size, H=seq_lens_sum, W=1), QKVs));
		}
		attn_output.push_back(prev_layer);
		lid_t res1;
		if(merge_post_processing(merge_mode)){
			res1 = add_to_mapping(n, out_proj_mapping, NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), attn_output);
		}
		else{
			res1 = n->add(NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), attn_output);
		}
		lid_t norm2;
		if(merge_post_processing(merge_mode)){
			norm2 = add_to_mapping(n, out_proj_mapping, NLAYER(layer_name+"_norm2", PTP, K=d_model, H=seq_lens_sum, W=1), {res1});
		}
		else{
			norm2 = n->add(NLAYER(layer_name+"_norm2", PTP, K=d_model, H=seq_lens_sum, W=1), {res1});
		}
		// Feed-Forward Network
	
		Network::layer_set ff1,ff2;
		auto ffn1_act_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_FFN1_Act");
		for(len_t j=0;j<d_ff/d_ff_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn1_tiling"+std::to_string(j);
			ff1.push_back(add_to_mapping(n, ffn1_act_mapping, NLAYER(name, Conv, C=d_model, K=d_ff_tiling_size, H=seq_lens_sum, W=1), {norm2}));
		}
		lid_t gelu = add_to_mapping(n, ffn1_act_mapping, NLAYER(layer_name+"_SwiGLU", PTP, K=d_ff, H=seq_lens_sum, W=1), ff1);
		auto ffn2_mapping = create_mapping_if_needed(
			n,
			merge_mode,
			merge_post_processing(merge_mode) ? layer_name+"_FFN2_Post" : layer_name+"_FFN2"
		);
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn2_tiling"+std::to_string(j);
			ff2.push_back(add_to_mapping(n, ffn2_mapping, NLAYER(name, Conv, C=d_ff, K=d_model_tiling_size, H=seq_lens_sum, W=1), {gelu}));
		}
		ff2.push_back(res1);
		lid_t res2;
		if(merge_post_processing(merge_mode)){
			res2 = add_to_mapping(n, ffn2_mapping, NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), ff2);
		}
		else{
			res2 = n->add(NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), ff2);
		}
		// lid_t ff1 = n->add(NLAYER(layer_name+"_ffn1", Conv, C=d_model, K=d_ff, H=seq_lens_sum, W=1), {norm1});
		// lid_t gelu = n->add(NLAYER(layer_name+"_GeLU", PTP, K=d_ff, H=seq_lens_sum, W=1), {ff1});
		// lid_t ff2 = n->add(NLAYER(layer_name+"_ffn2", Conv, C=d_ff, K=d_model, H=seq_lens_sum, W=1), {gelu});

		// // Residual connection after FFN
		// lid_t res2 = n->add(NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), {norm1, ff2});
		prev_layer = res2;
	}

	return n;
}

std::shared_ptr<Network> create_llama3_merged(
    const std::vector<Req>& reqs,
    len_t n_layers,
    len_t d_model,
    len_t n_heads,
    len_t d_head,
    len_t n_kv_heads,
    len_t d_ff,
    const std::string& mapping_merge_mode,
    len_t d_model_tiling_size,
    len_t d_ff_tiling_size
) {
	const std::string merge_mode = normalize_mapping_merge_mode(mapping_merge_mode);
	auto n=std::make_shared<Network>();
	assert(d_model==n_heads*d_head);
	assert(n_kv_heads <= n_heads && n_heads%n_kv_heads==0);
	if(d_model_tiling_size == 0){
		d_model_tiling_size = d_model;
	}
	if(d_ff_tiling_size == 0){
		d_ff_tiling_size = d_ff;
	}
	assert(d_model % d_model_tiling_size == 0);
	assert(d_ff % d_ff_tiling_size == 0);
	len_t seq_lens_sum = 0;
	for(auto& req : reqs){
		seq_lens_sum += req.lens;
	}

	InputData input("input", fmap_shape(d_model, seq_lens_sum, 1));
	InputData pos_encoding("pos_encoding", fmap_shape(d_model, seq_lens_sum, 1));

	lid_t prev_layer=n->add(NLAYER("input_add", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), {},0,{input,pos_encoding});
	const len_t kv_dim = n_kv_heads * d_head;

	for(len_t i=0;i<n_layers;i++){
		std::string layer_name="layer"+std::to_string(i);

		lid_t norm1 = n->add(NLAYER(layer_name+"_norm1", PTP, K=d_model, H=seq_lens_sum, W=1), {prev_layer});

		auto qkv_gen_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_QKV_Gen");
		lid_t K = add_to_mapping(n, qkv_gen_mapping, NLAYER(layer_name+"_k_gen_merged", Conv, C=d_model, K=kv_dim, H=seq_lens_sum, W=1), {norm1});
		n->getNode(K).mustWriteDRAM=true;
		lid_t V = add_to_mapping(n, qkv_gen_mapping, NLAYER(layer_name+"_v_gen_merged", Conv, C=d_model, K=kv_dim, H=seq_lens_sum, W=1), {norm1});
		n->getNode(V).mustWriteDRAM=true;
		lid_t Q = add_to_mapping(n, qkv_gen_mapping, NLAYER(layer_name+"_q_gen_merged", Conv, C=d_model, K=d_model, H=seq_lens_sum, W=1), {norm1});

		auto attn_qk_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_Attn_QK");
		Network::layer_set QKElts;
		for(len_t j=0;j<n_heads;j++)
		{
			std::string name = layer_name+"_head"+std::to_string(j);

			for(size_t k=0;k<reqs.size();k++)
			{
				const auto& req=reqs[k];
				std::string name_mul=name+"_req"+std::to_string(k);
				lid_t QK, QK_elt;
				if(req.type==Req::Type::Prefill||(req.type==Req::Type::ChunkedPrefill&&req.his_lens==0))
				{
					QK = add_to_mapping(n, attn_qk_mapping, NLAYER(name_mul+"_QK_P", Conv, C=d_head,K=req.lens, H=req.lens, W=1), {Q},0,{}, {K});
					QK_elt = add_to_mapping(n, attn_qk_mapping, NLAYER(name_mul+"_QK_elt_P", PTP, K=req.lens,H=req.lens, W=1), {QK});
				}
				else
				{
					InputData k_cache("k_cache", fmap_shape(req.his_lens, d_head, 1));
					QK = add_to_mapping(n, attn_qk_mapping, NLAYER(name_mul+"_QK_D", Conv, C=d_head,K=req.his_lens+req.lens, H=req.lens, W=1), {Q},0,{}, {K}, {k_cache});
					QK_elt = add_to_mapping(n, attn_qk_mapping, NLAYER(name_mul+"_QK_elt_D", PTP, K=req.his_lens+req.lens,H=req.lens, W=1), {QK});
				}
				QKElts.push_back(QK_elt);
			}
		}

		auto attn_av_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_Attn_AV");
		Network::layer_set QKVs;
		size_t qk_elt_idx = 0;
		for(len_t j=0;j<n_heads;j++)
		{
			std::string name = layer_name+"_head"+std::to_string(j);
			lid_t QKV;

			for(size_t k=0;k<reqs.size();k++)
			{
				const auto& req=reqs[k];
				std::string name_mul=name+"_req"+std::to_string(k);
				const lid_t QK_elt = QKElts[qk_elt_idx++];
				if(req.type==Req::Type::Prefill||(req.type==Req::Type::ChunkedPrefill&&req.his_lens==0))
				{
					QKV = add_to_mapping(n, attn_av_mapping, NLAYER(name_mul+"_QKV_P", Conv, C=req.lens,K=d_head, H=req.lens, W=1), {QK_elt},0,{}, {V});
				}
				else if(req.type==Req::Type::Decode||(req.type==Req::Type::ChunkedPrefill&&req.his_lens!=0))
				{
					InputData v_cache("v_cache", fmap_shape(req.his_lens, d_head, 1));
					QKV = add_to_mapping(n, attn_av_mapping, NLAYER(name_mul+"_QKV_D", Conv, C=req.his_lens+req.lens,K=d_head, H=req.lens, W=1), {QK_elt},0,{}, {V}, {v_cache});
				}
				QKVs.push_back(QKV);
			}
		}

		auto out_proj_mapping = create_mapping_if_needed(
			n,
			merge_mode,
			merge_post_processing(merge_mode) ? layer_name+"_Out_Proj_Post" : layer_name+"_Out_Proj"
		);
		Network::layer_set attn_output;
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_out_proj_tiling_"+std::to_string(j);
			attn_output.push_back(add_to_mapping(n, out_proj_mapping, NLAYER(name, Conv, C=d_model, K=d_model_tiling_size, H=seq_lens_sum, W=1), QKVs));
		}
		lid_t res1;
		if(merge_post_processing(merge_mode)){
			attn_output.push_back(prev_layer);
			res1 = add_to_mapping(n, out_proj_mapping, NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), attn_output);
		}
		else{
			attn_output.push_back(prev_layer);
			res1 = n->add(NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), attn_output);
		}
		lid_t norm2;
		if(merge_post_processing(merge_mode)){
			norm2 = add_to_mapping(n, out_proj_mapping, NLAYER(layer_name+"_norm2", PTP, K=d_model, H=seq_lens_sum, W=1), {res1});
		}
		else{
			norm2 = n->add(NLAYER(layer_name+"_norm2", PTP, K=d_model, H=seq_lens_sum, W=1), {res1});
		}

		auto ffn1_act_mapping = create_mapping_if_needed(n, merge_mode, layer_name+"_FFN1_Act");
		Network::layer_set ff1,ff2;
		for(len_t j=0;j<d_ff/d_ff_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn1_tiling"+std::to_string(j);
			ff1.push_back(add_to_mapping(n, ffn1_act_mapping, NLAYER(name, Conv, C=d_model, K=d_ff_tiling_size, H=seq_lens_sum, W=1), {norm2}));
		}
		lid_t swiglu = add_to_mapping(n, ffn1_act_mapping, NLAYER(layer_name+"_SwiGLU", PTP, K=d_ff, H=seq_lens_sum, W=1), ff1);
		auto ffn2_mapping = create_mapping_if_needed(
			n,
			merge_mode,
			merge_post_processing(merge_mode) ? layer_name+"_FFN2_Post" : layer_name+"_FFN2"
		);
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn2_tiling"+std::to_string(j);
			ff2.push_back(add_to_mapping(n, ffn2_mapping, NLAYER(name, Conv, C=d_ff, K=d_model_tiling_size, H=seq_lens_sum, W=1), {swiglu}));
		}
		lid_t res2;
		if(merge_post_processing(merge_mode)){
			ff2.push_back(res1);
			res2 = add_to_mapping(n, ffn2_mapping, NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), ff2);
		}
		else{
			ff2.push_back(res1);
			res2 = n->add(NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), ff2);
		}
		prev_layer = res2;
	}

	return n;
}
