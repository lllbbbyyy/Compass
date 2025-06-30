#include "network.h"

#include <cassert>

#include "util.h"
#include "compass/request_generator.h"
#include "debug.h"
#include <vector>
#include <string>
#include <memory>

std::shared_ptr<Network> create_GPT3(const std::vector<Req>& reqs,len_t n_layers,len_t d_model,len_t n_heads,len_t d_head,len_t d_ff,len_t d_model_tiling_size,len_t d_ff_tiling_size)
{
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
		Network::layer_set QKVs;
		for(len_t j=0;j<n_heads;j++)
		{
			// Network::lid_t Q, K, V, newK, newKt, Kt, newV, newVt, Vt,  QK, QK_elt, QKV;
			std::string name = layer_name+"_head"+std::to_string(j);

			lid_t Q, K, V,  QK, QK_elt, QKV;

			//DEBUG("seq_lens_sum",seq_lens_sum);
			Q = n->add(NLAYER(name+"_Q", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {prev_layer});
			K = n->add(NLAYER(name+"_K", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {prev_layer});
			n->getNode(K).mustWriteDRAM=true;
			V = n->add(NLAYER(name+"_V", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {prev_layer});
			n->getNode(V).mustWriteDRAM=true;

			//split by req
			for(size_t k=0;k<reqs.size();k++)
			{
				const auto& req=reqs[k];
				std::string name_mul=name+"_req"+std::to_string(k);
				if(req.type==Req::Type::Prefill)
				{
					QK = n->add(NLAYER(name_mul+"_QK_P", Conv, C=d_head,K=req.lens, H=req.lens, W=1), {Q},0,{}, {K});
					QK_elt = n->add(NLAYER(name_mul+"_QK_elt_P", PTP, K=req.lens,H=req.lens, W=1), {QK});
					QKV = n->add(NLAYER(name_mul+"_QKV_P", Conv, C=req.lens,K=d_head, H=req.lens, W=1), {QK_elt},0,{}, {V});
				}
				else if(req.type==Req::Type::Decode)
				{
					// all K and V from DRAM after calculating newK and newV
					InputData k_cache("k_cache", fmap_shape(req.his_lens, d_head, 1)); // transposed K cache
					InputData v_cache("v_cache", fmap_shape(req.his_lens, d_head, 1)); // V cache
					QK = n->add(NLAYER(name_mul+"_QK_D", Conv, C=d_head,K=req.his_lens+1, H=1, W=1), {Q},0,{}, {K}, {k_cache});
					QK_elt = n->add(NLAYER(name_mul+"_QK_elt_D", PTP, K=1*(req.his_lens+1),H=1, W=1), {QK});
					QKV = n->add(NLAYER(name_mul+"_QKV_D", Conv, C=req.his_lens+1,K=d_head, H=1, W=1), {QK_elt},0,{}, {V}, {v_cache});
				}
				QKVs.push_back(QKV);
			}

		}
		
		// lid_t attn_output = n->add(NLAYER(layer_name + "_out_proj", Conv, C=d_model, H=seq_lens_sum, W=1), QKVs);
		assert(d_ff % d_ff_tiling_size == 0); // Ensure d_ff is divisible by ffn_tiling_size
		assert(d_model % d_model_tiling_size == 0); // Ensure d_model is divisible by ffn_tiling_size

		Network::layer_set attn_output;
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_out_proj_tiling_"+std::to_string(j);
			attn_output.push_back(n->add(NLAYER(name, Conv, C=d_model, K=d_model_tiling_size, H=seq_lens_sum, W=1), QKVs));
		}
		attn_output.push_back(prev_layer);
		lid_t res1 = n->add(NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), attn_output);

		// // Residual connection after attention
		// lid_t res1 = n->add(NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), {prev_layer, attn_output});
		
		// LayerNorm after first residual
		lid_t norm1 = n->add(NLAYER(layer_name+"_norm1", PTP, K=d_model, H=seq_lens_sum, W=1), {res1});
		// Feed-Forward Network
	
		Network::layer_set ff1,ff2;
		for(len_t j=0;j<d_ff/d_ff_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn1_tiling"+std::to_string(j);
			ff1.push_back(n->add(NLAYER(name, Conv, C=d_model, K=d_ff_tiling_size, H=seq_lens_sum, W=1), {norm1}));
		}
		lid_t gelu = n->add(NLAYER(layer_name+"_GeLU", PTP, K=d_ff, H=seq_lens_sum, W=1), ff1);
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn2_tiling"+std::to_string(j);
			ff2.push_back(n->add(NLAYER(name, Conv, C=d_ff, K=d_model_tiling_size, H=seq_lens_sum, W=1), {gelu}));
		}
		ff2.push_back(norm1);
		lid_t res2 = n->add(NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), ff2);
		lid_t norm2 = n->add(NLAYER(layer_name+"_norm2", PTP, K=d_model, H=seq_lens_sum, W=1), {res2});
		// lid_t ff1 = n->add(NLAYER(layer_name+"_ffn1", Conv, C=d_model, K=d_ff, H=seq_lens_sum, W=1), {norm1});
		// lid_t gelu = n->add(NLAYER(layer_name+"_GeLU", PTP, K=d_ff, H=seq_lens_sum, W=1), {ff1});
		// lid_t ff2 = n->add(NLAYER(layer_name+"_ffn2", Conv, C=d_ff, K=d_model, H=seq_lens_sum, W=1), {gelu});

		// // Residual connection after FFN
		// lid_t res2 = n->add(NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), {norm1, ff2});
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
    len_t d_ff_tiling_size
) {
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

		Network::layer_set QKVs;
		lid_t Q, K=0, V=0,  QK, QK_elt, QKV;
		for(len_t j=0;j<n_heads;j++)
		{
			// Network::lid_t Q, K, V, newK, newKt, Kt, newV, newVt, Vt,  QK, QK_elt, QKV;
			std::string name = layer_name+"_head"+std::to_string(j);

			//DEBUG("seq_lens_sum",seq_lens_sum);
			Q = n->add(NLAYER(name+"_Q", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {norm1});
			if(j%(n_heads/n_kv_heads)==0) // Only the first head in each kv group computes K and V
			{
				K = n->add(NLAYER(name+"_K", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {norm1});
				n->getNode(K).mustWriteDRAM=true;
				V = n->add(NLAYER(name+"_V", Conv, C=d_model,K=d_head, H=seq_lens_sum, W=1), {norm1});
				n->getNode(V).mustWriteDRAM=true;
			}
			//split by req
			for(size_t k=0;k<reqs.size();k++)
			{
				const auto& req=reqs[k];
				std::string name_mul=name+"_req"+std::to_string(k);
				if(req.type==Req::Type::Prefill)
				{
					QK = n->add(NLAYER(name_mul+"_QK_P", Conv, C=d_head,K=req.lens, H=req.lens, W=1), {Q},0,{}, {K});
					QK_elt = n->add(NLAYER(name_mul+"_QK_elt_P", PTP, K=req.lens,H=req.lens, W=1), {QK});
					QKV = n->add(NLAYER(name_mul+"_QKV_P", Conv, C=req.lens,K=d_head, H=req.lens, W=1), {QK_elt},0,{}, {V});
				}
				else if(req.type==Req::Type::Decode)
				{
					// all K and V from DRAM after calculating newK and newV
					InputData k_cache("k_cache", fmap_shape(req.his_lens, d_head, 1)); // transposed K cache
					InputData v_cache("v_cache", fmap_shape(req.his_lens, d_head, 1)); // V cache
					QK = n->add(NLAYER(name_mul+"_QK_D", Conv, C=d_head,K=req.his_lens+1, H=1, W=1), {Q},0,{}, {K}, {k_cache});
					QK_elt = n->add(NLAYER(name_mul+"_QK_elt_D", PTP, K=1*(req.his_lens+1),H=1, W=1), {QK});
					QKV = n->add(NLAYER(name_mul+"_QKV_D", Conv, C=req.his_lens+1,K=d_head, H=1, W=1), {QK_elt},0,{}, {V}, {v_cache});
				}
				QKVs.push_back(QKV);
			}

		}
		
		// lid_t attn_output = n->add(NLAYER(layer_name + "_out_proj", Conv, C=d_model, H=seq_lens_sum, W=1), QKVs);
		assert(d_ff % d_ff_tiling_size == 0); // Ensure d_ff is divisible by ffn_tiling_size
		assert(d_model % d_model_tiling_size == 0); // Ensure d_model is divisible by ffn_tiling_size

		Network::layer_set attn_output;
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_out_proj_tiling_"+std::to_string(j);
			attn_output.push_back(n->add(NLAYER(name, Conv, C=d_model, K=d_model_tiling_size, H=seq_lens_sum, W=1), QKVs));
		}
		attn_output.push_back(prev_layer);
		lid_t res1 = n->add(NLAYER(layer_name+"_res1", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), attn_output);
		lid_t norm2 = n->add(NLAYER(layer_name+"_norm2", PTP, K=d_model, H=seq_lens_sum, W=1), {res1});
		// Feed-Forward Network
	
		Network::layer_set ff1,ff2;
		for(len_t j=0;j<d_ff/d_ff_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn1_tiling"+std::to_string(j);
			ff1.push_back(n->add(NLAYER(name, Conv, C=d_model, K=d_ff_tiling_size, H=seq_lens_sum, W=1), {norm2}));
		}
		lid_t gelu = n->add(NLAYER(layer_name+"_SwiGLU", PTP, K=d_ff, H=seq_lens_sum, W=1), ff1);
		for(len_t j=0;j<d_model/d_model_tiling_size;j++)
		{
			std::string name = layer_name+"_ffn2_tiling"+std::to_string(j);
			ff2.push_back(n->add(NLAYER(name, Conv, C=d_ff, K=d_model_tiling_size, H=seq_lens_sum, W=1), {gelu}));
		}
		ff2.push_back(res1);
		lid_t res2 = n->add(NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), ff2);
		// lid_t ff1 = n->add(NLAYER(layer_name+"_ffn1", Conv, C=d_model, K=d_ff, H=seq_lens_sum, W=1), {norm1});
		// lid_t gelu = n->add(NLAYER(layer_name+"_GeLU", PTP, K=d_ff, H=seq_lens_sum, W=1), {ff1});
		// lid_t ff2 = n->add(NLAYER(layer_name+"_ffn2", Conv, C=d_ff, K=d_model, H=seq_lens_sum, W=1), {gelu});

		// // Residual connection after FFN
		// lid_t res2 = n->add(NLAYER(layer_name+"_res2", Eltwise, K=d_model, H=seq_lens_sum, W=1, N=2), {norm1, ff2});
		prev_layer = res2;
	}

	return n;
}