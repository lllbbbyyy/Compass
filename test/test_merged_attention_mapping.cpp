#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "nns/nns.h"

namespace {
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
}

int main()
{
	const std::vector<Req> reqs = {
		Req(0, Req::Type::Prefill, 16, 0),
		Req(1, Req::Type::Decode, 1, 31),
	};

	check_attention_mapping(create_GPT3_merged(reqs, 1, 128, 2, 64, 512, "stage"));
	check_attention_mapping(create_llama3_merged(reqs, 1, 128, 2, 64, 1, 512, "stage"));

	std::cout << "Merged attention mapping test passed." << std::endl;
	return 0;
}
