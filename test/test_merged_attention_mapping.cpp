#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "nns/nns.h"

namespace {
const Network::MappingNode* find_mapping(
	const std::shared_ptr<Network>& model,
	const std::string& name
)
{
	for(size_t mapping_id = 0; mapping_id < model->mapping_len(); ++mapping_id){
		const auto& mapping = model->getMappingNode(static_cast<Network::mapping_id_t>(mapping_id));
		if(mapping.name == name){
			return &mapping;
		}
	}
	return nullptr;
}

std::vector<lid_t> collect_single_layer_mappings(
	const std::shared_ptr<Network>& model,
	const std::string& prefix
)
{
	std::vector<lid_t> exec_ids;
	for(size_t mapping_id = 0; mapping_id < model->mapping_len(); ++mapping_id){
		const auto& mapping = model->getMappingNode(static_cast<Network::mapping_id_t>(mapping_id));
		if(mapping.name.rfind(prefix, 0) != 0){
			continue;
		}
		assert(mapping.exec_layer_ids.size() == 1);
		exec_ids.push_back(mapping.exec_layer_ids.front());
	}
	return exec_ids;
}

void assert_depends_on_all(
	const std::shared_ptr<Network>& model,
	lid_t consumer,
	const std::vector<lid_t>& producers
)
{
	const auto& prevs = model->getNode(consumer).getPrevs();
	for(lid_t producer : producers){
		assert(std::find(prevs.begin(), prevs.end(), producer) != prevs.end());
	}
}

void check_attention_mapping(const std::shared_ptr<Network>& model)
{
	const Network::MappingNode* qk_mapping = nullptr;
	const Network::MappingNode* av_mapping = nullptr;

	for(size_t mapping_id = 0; mapping_id < model->mapping_len(); ++mapping_id){
		const auto& mapping = model->getMappingNode(static_cast<Network::mapping_id_t>(mapping_id));
		if(mapping.name == "layer0_Attn_QK"){
			qk_mapping = &mapping;
		}
		else if(mapping.name == "layer0_Attn_AV"){
			av_mapping = &mapping;
		}
	}

	assert(qk_mapping != nullptr);
	assert(av_mapping != nullptr);
	assert(!qk_mapping->exec_layer_ids.empty());
	assert(!av_mapping->exec_layer_ids.empty());
	assert(qk_mapping->exec_layer_ids.back() < av_mapping->exec_layer_ids.front());
	assert(qk_mapping->exec_layer_ids.size() == 8);
	assert(av_mapping->exec_layer_ids.size() == 4);

	size_t softmax_count = 0;
	for(size_t i = 1; i < qk_mapping->exec_layer_ids.size(); ++i){
		assert(qk_mapping->exec_layer_ids[i] == qk_mapping->exec_layer_ids[i - 1] + 1);
	}
	for(lid_t exec_id : qk_mapping->exec_layer_ids){
		const std::string& name = model->getNode(exec_id).name();
		assert(name.find("_QK_") != std::string::npos);
		if(name.find("_QK_elt_") != std::string::npos){
			++softmax_count;
		}
	}
	assert(softmax_count == 4);

	for(size_t i = 1; i < av_mapping->exec_layer_ids.size(); ++i){
		assert(av_mapping->exec_layer_ids[i] == av_mapping->exec_layer_ids[i - 1] + 1);
	}
	for(lid_t exec_id : av_mapping->exec_layer_ids){
		const std::string& name = model->getNode(exec_id).name();
		assert(name.find("_QKV_") != std::string::npos);
		assert(name.find("_QK_elt_") == std::string::npos);
	}
}

void check_free_tensor_parallel_mapping(
	const std::shared_ptr<Network>& model,
	bool llama
)
{
	const auto out_proj = collect_single_layer_mappings(model, "layer0_out_proj_tiling_");
	const auto ffn1 = collect_single_layer_mappings(model, "layer0_ffn1_tiling");
	const auto ffn2 = collect_single_layer_mappings(model, "layer0_ffn2_tiling");
	assert(out_proj.size() == 4);
	assert(ffn1.size() == 4);
	assert(ffn2.size() == 4);

	for(const auto& shards : {out_proj, ffn1, ffn2}){
		for(size_t i = 1; i < shards.size(); ++i){
			assert(model->mapping_id_for_exec(shards[i - 1]) != model->mapping_id_for_exec(shards[i]));
		}
	}

	const auto* out_post = find_mapping(model, "layer0_Out_Proj_Post");
	const auto* ffn1_act = find_mapping(model, "layer0_FFN1_Act");
	const auto* ffn2_post = find_mapping(model, "layer0_FFN2_Post");
	assert(out_post != nullptr && out_post->exec_layer_ids.size() == 2);
	assert(ffn1_act != nullptr && ffn1_act->exec_layer_ids.size() == 1);
	assert(ffn2_post != nullptr && ffn2_post->exec_layer_ids.size() == (llama ? 1 : 2));
	assert(model->mapping_id_for_exec(out_proj.back()) < model->mapping_id_for_exec(out_post->exec_layer_ids.front()));
	assert(model->mapping_id_for_exec(ffn1.back()) < model->mapping_id_for_exec(ffn1_act->exec_layer_ids.front()));
	assert(model->mapping_id_for_exec(ffn2.back()) < model->mapping_id_for_exec(ffn2_post->exec_layer_ids.front()));

	assert_depends_on_all(model, out_post->exec_layer_ids.front(), out_proj);
	assert_depends_on_all(model, ffn1_act->exec_layer_ids.front(), ffn1);
	assert_depends_on_all(model, ffn2_post->exec_layer_ids.front(), ffn2);
}

void check_unsplit_stage_post_mapping(const std::shared_ptr<Network>& model)
{
	const auto* out_post = find_mapping(model, "layer0_Out_Proj_Post");
	const auto* ffn1_act = find_mapping(model, "layer0_FFN1_Act");
	const auto* ffn2_post = find_mapping(model, "layer0_FFN2_Post");
	assert(out_post != nullptr && out_post->exec_layer_ids.size() == 3);
	assert(ffn1_act != nullptr && ffn1_act->exec_layer_ids.size() == 2);
	assert(ffn2_post != nullptr && ffn2_post->exec_layer_ids.size() == 3);
}
}

int main()
{
	const std::vector<Req> reqs = {
		Req(0, Req::Type::Prefill, 16, 0),
		Req(1, Req::Type::Decode, 1, 31),
	};

	check_attention_mapping(create_GPT3_merged(reqs, 1, 128, 2, 64, 512, "stage"));
	check_attention_mapping(create_llama3_merged(reqs, 1, 128, 2, 64, 1, 512, "stage"));
	check_unsplit_stage_post_mapping(create_GPT3_merged(reqs, 1, 128, 2, 64, 512, "stage_post"));
	check_free_tensor_parallel_mapping(
		create_GPT3_merged(reqs, 1, 128, 2, 64, 512, "stage_post", 32, 128),
		false
	);
	check_free_tensor_parallel_mapping(
		create_llama3_merged(reqs, 1, 128, 2, 64, 1, 512, "stage_post", 32, 128),
		true
	);

	std::cout << "Merged attention mapping test passed." << std::endl;
	return 0;
}
