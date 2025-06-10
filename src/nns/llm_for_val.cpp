#include "network.h"

#include <cassert>

#include "util.h"
#include "compass/request_generator.h"
#include "debug.h"
#include <vector>
#include <string>
#include <memory>

std::shared_ptr<Network> create_LLM_block_decode(len_t num_heads, len_t d_head, len_t kv_head_num, len_t seq_len)
{
	assert(seq_len >= 1);
	assert(kv_head_num >= 1 && num_heads % kv_head_num == 0);
	const len_t d_model = num_heads * d_head;
	auto n=std::make_shared<Network>();
	// Input layer for the new token
	InputData input_q("input_q", fmap_shape(num_heads * d_head, 1, 1));
	
	// LayerNorm on the new token
	lid_t norm1 = n->add(NLAYER("norm1", PTP, K=num_heads * d_head, H=1, W=1), {}, 0, {input_q});

	// Attention with KV-cache
    lid_t attn_output;
	{
		// lid_t Q, K, V, newK, newKt, Kt, newV, newVt, Vt,  QK, QK_elt, QKV;
		lid_t Q, K, V, newK, newK_one_head, newKt_one_head, newV,  QK, QK_elt, QKV;
		Network::layer_set newKs;
		const std::string name = "attention";

		Q = n->add(NLAYER(name + "_Q", Conv, C=d_model, H=1, W=1), {norm1});

		for(len_t i = 0; i < kv_head_num; ++i){
			newK_one_head = n->add(NLAYER(name + "_newK_head_" + std::to_string(i), Conv, C=d_model, K=d_head, H=1, W=1), {norm1});
			newKt_one_head = n->add(NLAYER(name + "_newKt_head_" + std::to_string(i), Transpose, K=1, H=d_head, W=1), {newK_one_head});
			newKs.push_back(newKt_one_head);
		}
		newK = n->add(NLAYER(name + "_newK", PTP, K=kv_head_num, H=d_head, W=1), newKs); // concat only on C channel
		newV = n->add(NLAYER(name + "_newV", Conv, C=d_model, K=kv_head_num*d_head, H=1, W=1), {norm1});
		
		// all K and V from DRAM after calculating newK and newV
		InputData k_cache("k_cache", fmap_shape(kv_head_num*(seq_len-1), d_head, 1)); // transposed K cache
		InputData v_cache("v_cache", fmap_shape(seq_len-1, kv_head_num*d_head, 1)); // V cache

		if (kv_head_num != num_heads) {
			lid_t newK_expand, newV_expand, k_cache_expand, v_cache_expand, Vt, newV_expand_t;
			newK_expand = n->add(NLAYER(name + "_newK_expand", Upsample, K=num_heads, H=d_head, W=1, sK=num_heads/kv_head_num, sH=1, sW=1), {newK});
			k_cache_expand = n->add(NLAYER(name + "_k_cache_expand", Upsample, K=num_heads*(seq_len-1), H=d_head, W=1, sK=num_heads/kv_head_num, sH=1, sW=1), {}, 0, {k_cache});
			K = n->add(NLAYER(name + "_K", PTP, K=num_heads*seq_len, H=d_head, W=1), {newK_expand, k_cache_expand});

			newV_expand = n->add(NLAYER(name + "_newV_expand", Upsample, K=num_heads*d_head, H=1, W=1, sK=num_heads/kv_head_num, sH=1, sW=1), {newV});
			newV_expand_t = n->add(NLAYER(name + "_newV_expand_t", Transpose, K=1, H=num_heads*d_head, W=1), {newV_expand});
			v_cache_expand = n->add(NLAYER(name + "_v_cache_expand", Upsample, K=seq_len-1, H=num_heads*d_head, W=1, sK=1, sH=num_heads/kv_head_num, sW=1), {}, 0, {v_cache});
			Vt = n->add(NLAYER(name + "_Vt", PTP, K=seq_len, H=num_heads*d_head, W=1), {newV_expand_t, v_cache_expand});
			V = n->add(NLAYER(name + "_V", Transpose, K=num_heads*d_head, H=seq_len, W=1), {Vt});
		} else {
			lid_t newVt, Vt; 
			K = n->add(NLAYER(name + "_K", PTP, K=num_heads*seq_len, H=d_head, W=1), {newK}, 0, {k_cache});
			newVt = n->add(NLAYER(name + "_newVt", Transpose, K=1, H=kv_head_num*d_head, W=1), {newV});
			Vt = n->add(NLAYER(name + "_Vt_concat", PTP, K=seq_len, H=kv_head_num*d_head, W=1), {newVt}, 0, {v_cache});
			V = n->add(NLAYER(name + "_V", Transpose, K=kv_head_num*d_head, H=seq_len, W=1), {Vt});
		}
		QK = n->add(NLAYER(name + "_QK", GroupConv, H=1, W=1, C=num_heads*d_head, K=num_heads*seq_len, G=num_heads), {Q}, 0, {}, {K}); // K_cache is the weight(from DRAM) of this layer
		QK_elt = n->add(NLAYER(name + "_QK_elt", PTP, K=num_heads*seq_len, H=1, W=1), {QK});
		QKV = n->add(NLAYER(name + "_QKV", GroupConv, H=1, W=1, C=num_heads*seq_len, K=num_heads*d_head, G=num_heads), {QK_elt}, 0, {}, {V}); // V_cache is the weight(from DRAM) of this layer
		// Output projection
		attn_output = n->add(NLAYER(name + "_Out_proj", Conv, C=d_model, H=1, W=1), {QKV});
	}
	// Residual connection after attention
    lid_t res1 = n->add(NLAYER("res1", Eltwise, K=num_heads * d_head, H=1, W=1, N=2), {attn_output}, 0, {input_q});
	// lid_t res1 = n->add(NLAYER("res1", Eltwise, K=num_heads * d_head, H=1, W=1, N=2), {attn_output, norm1});
	// LayerNorm after first residual
    lid_t norm2 = n->add(NLAYER("norm2", PTP, K=num_heads * d_head, H=1, W=1), {res1});
    // Feed-Forward Network
	const len_t d_ff = 4 * d_model;
    lid_t ff1 = n->add(NLAYER("ffn1", Conv, C=d_model, K=d_ff, H=1, W=1), {norm2});
    lid_t ff2 = n->add(NLAYER("GeLU", PTP, K=d_ff, H=1, W=1), {ff1});
    lid_t ff3 = n->add(NLAYER("ffn2", Conv, C=d_ff, K=d_model, H=1, W=1), {ff2});

    // Residual connection after FFN
    n->add(NLAYER("res2", Eltwise, K=d_model, H=1, W=1, N=2), {res1, ff3});
    // for(lid_t i=0;i<n->len();i++){
    //     n->getNode(i).mustWriteDRAM=true;
    // }
    // for(lid_t i=0;i<n->len();i++){
    //     n->getNode(i).mustReadDRAM=true;
    // }

	return n;
};

lid_t add_attention(
		std::shared_ptr<Network> n, const std::string& name,
		len_t len, len_t numG, len_t gSize,
		lid_t prevQ, lid_t prevK, lid_t prevV, len_t kv_head_num=1){


	lid_t Q, K, Kt, V, QK, QK_elt, QKV;
	if (kv_head_num == 1) {
		Network::layer_set Ks;
		Q = n->add(NLAYER(name + "_Q", Conv, H=len, W=1, C=numG*gSize), {prevQ});
		for(len_t i = 0; i < numG; ++i){
			K = n->add(NLAYER(name + "_K" + std::to_string(i), Conv, C=numG*gSize, K=gSize, H=len, W=1), {prevK});
			Kt = n->add(NLAYER(name + "_Kt" + std::to_string(i), Transpose, K=len, H=gSize, W=1), {K});
			Ks.push_back(Kt);
		}
		K = n->add(NLAYER(name + "_K", PTP, K=numG*len, H=gSize, W=1), Ks);
		V = n->add(NLAYER(name + "_V", Conv, C=numG*gSize, H=len, W=1), {prevV});
		QK = n->add(NLAYER(name + "_QK", GroupConv, H=len, W=1, C=numG*gSize, K=numG*len, G=numG), {Q}, 0, {}, {K});
		QK_elt = n->add(NLAYER(name + "_QK_elt", PTP, K=numG*len, H=len, W=1), {QK});
		QKV = n->add(NLAYER(name + "_QKV", GroupConv, H=len, W=1, C=numG*len, K=numG*gSize, G=numG), {QK_elt}, 0, {}, {V});
		return n->add(NLAYER(name + "_FC", Conv, H=len, W=1, C=numG*gSize), {QKV});
	} else { 
		/*
		// too many layers
		Network::layer_set QKVs;
		// number of K, V == kv_head_num
		for(len_t i = 0; i < kv_head_num; ++i){
			const std::string suffix = "_group" + std::to_string(i);
			K = n->add(NLAYER(name + "_K" + suffix, Conv, C=numG*gSize, K=gSize, H=len, W=1), {prevK});
			Kt = n->add(NLAYER(name + "_Kt" + suffix, Transpose, K=len, H=gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {K});
			V = n->add(NLAYER(name + "_V" + suffix, Conv, C=numG*gSize, K=gSize, H=len, W=1), {prevV});
			for (len_t j = 0; j < numG/kv_head_num; ++j) {
				Q = n->add(NLAYER(name + "_Q" + suffix, Conv, H=len, W=1, C=numG*gSize, K=gSize), {prevQ});
				QK = n->add(NLAYER(name + "_QK" + suffix, Conv, H=len, W=1, C=gSize, K=len), {Q}, 0, {}, {Kt});
				QK_elt = n->add(NLAYER(name + "_QK_elt" + suffix, PTP, K=len, H=len, W=1), {QK});
				QKV = n->add(NLAYER(name + "_QKV" + suffix, Conv, H=len, W=1, C=len, K=gSize), {QK_elt}, 0, {}, {V});
				QKVs.push_back(QKV);
			}
		}
		return n->add(NLAYER(name + "_FC", Conv, H=len, W=1, C=numG*gSize), QKVs);
		*/
		// number of K, V == kv_head_num
		Network::layer_set Ks;
		Q = n->add(NLAYER(name + "_Q", Conv, H=len, W=1, C=numG*gSize), {prevQ});
		for(len_t i = 0; i < kv_head_num; ++i){
			K = n->add(NLAYER(name + "_K" + std::to_string(i), Conv, C=numG*gSize, K=gSize, H=len, W=1), {prevK});
			Kt = n->add(NLAYER(name + "_Kt" + std::to_string(i), Transpose, K=len, H=gSize, W=1), {K});
			Ks.push_back(Kt);
		}
		K = n->add(NLAYER(name + "_K", PTP, K=kv_head_num*len, H=gSize, W=1), Ks);
		// K_expand = n->add(NLAYER(name + "_K_expand", Upsample, K=numG*len, H=gSize, W=1, sK=numG/kv_head_num, sH=1, sW=1), {K});
		// K_expand = n->add(NLAYER(name + "_K_expand", PTP, K=numG*len, H=gSize, W=1), {K});
		V = n->add(NLAYER(name + "_V", Conv, C=numG*gSize, K = kv_head_num*gSize, H=len, W=1), {prevV});
		// V_expand = n->add(NLAYER(name + "_V_expand", Upsample, K=numG*gSize, H=len, W=1, sK=numG/kv_head_num, sH=1, sW=1), {V});
		// V_expand = n->add(NLAYER(name + "_V_expand", PTP, K=numG*gSize, H=len, W=1), {V});
		QK = n->add(NLAYER(name + "_QK", GroupConv, H=len, W=1, C=numG*gSize, K=numG*len, G=numG), {Q}, 0, {}, {K});
		QK_elt = n->add(NLAYER(name + "_QK_elt", PTP, K=numG*len, H=len, W=1), {QK});
		QKV = n->add(NLAYER(name + "_QKV", GroupConv, H=len, W=1, C=numG*len, K=numG*gSize, G=numG), {QK_elt}, 0, {}, {V});
		return n->add(NLAYER(name + "_FC", Conv, H=len, W=1, C=numG*gSize), {QKV});
	}	
}


std::shared_ptr<Network> create_LLM_block_prefill(len_t num_heads, len_t d_head, len_t group_num, len_t seq_len)
{
	assert(seq_len >= 1);
	assert(group_num >= 1 && num_heads % group_num == 0);
	const len_t d_model = num_heads * d_head;
	auto n=std::make_shared<Network>();
	// Input layer for the new token
	InputData input_qkv("input_qkv", fmap_shape(num_heads * d_head, seq_len, 1));
	
	// LayerNorm on the new token
	lid_t norm1 = n->add(NLAYER("norm1", PTP, K=num_heads * d_head, H=seq_len, W=1), {}, 0, {input_qkv});

	// Attention with KV-cache
    lid_t attn_output = add_attention(n, "attention", seq_len, num_heads, d_head, norm1, norm1, norm1,  group_num);
	// Residual connection after attention
    lid_t res1 = n->add(NLAYER("res1", Eltwise, K=d_model, H=seq_len, W=1, N=2), {attn_output}, 0, {input_qkv});
	//lid_t res1 = n->add(NLAYER("res1", Eltwise, K=d_model, H=seq_len, W=1, N=2), {attn_output, norm1});
	// LayerNorm after first residual
    lid_t norm2 = n->add(NLAYER("norm2", PTP, K=num_heads * d_head, H=seq_len, W=1), {res1});
    // Feed-Forward Network
	const len_t d_ff = 4 * d_model;
    lid_t ff1 = n->add(NLAYER("ffn1", Conv, C=d_model, K=d_ff, H=seq_len, W=1), {norm2});
    lid_t ff2 = n->add(NLAYER("GeLU", PTP, K=d_ff, H=seq_len, W=1), {ff1});
    lid_t ff3 = n->add(NLAYER("ffn2", Conv, C=d_ff, K=d_model, H=seq_len, W=1), {ff2});

    // Residual connection after FFN
    n->add(NLAYER("res2", Eltwise, K=d_model, H=seq_len, W=1, N=2), {res1, ff3});

	return n;
};
