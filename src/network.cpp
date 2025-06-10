#include "network.h"

#include <cassert>
#include <stdexcept>

#include "core_mapping.h"
#include "debug.h"

InputData::InputData(const std::string& _name, const fmap_shape& _data_shape,int dram_id)
	:name(_name), data_shape(_data_shape),DRAM_id(dram_id)
{
	assert(data_shape.size > 0);
}

const fmap_shape& InputData::get_shape() const{
	return data_shape;
}

void Node::reset()
{
	// DEBUG("Node::reset", l->get_name(),isTiling,ctile,ktile,htile);
	isTiling=false;
	ctile=1;
	ktile=1;
	htile=1;
}

Node::Node(Layer *_l, const std::vector<lid_t> &_ifmPrevs, len_t _external_C, bwidth_t width, const std::vector<lid_t> &_wgtPrevs) : l(_l)
{
	l->ifmPrevs=_ifmPrevs;
	l->wgtPrevs=_wgtPrevs;

	std::vector<int> temp = _ifmPrevs;
	temp.insert(temp.end(), _wgtPrevs.begin(), _wgtPrevs.end());
	l->prevs=temp;
	l->external_C=_external_C;
	if(width > 0) l->set_bitwidth(width);
}

Layer& Node::layer() const{
	return *l;
}

const std::string& Node::name() const{
	return l->get_name();
}

const std::vector<lid_t>& Node::getIfmPrevs() const{
	return l->ifmPrevs;
}

const std::vector<lid_t>& Node::getWgtPrevs() const{
	return l->wgtPrevs;
}

const std::vector<lid_t>& Node::getPrevs() const{
	return l->prevs;
}

const std::vector<lid_t>& Node::get_nexts() const{
	return l->nexts;
}

utime_t Node::get_utime() const{
	return l->get_utime();
}

len_t Node::get_external_C() const{
	return l->external_C;
}

bool Node::hasWgtPrevs() const{
	return l->wgtPrevs.size() > 0;
}

void Node::add_next(lid_t n){
	l->nexts.push_back(n);
}


Network::Network(){}

void Network::err_mismatch(const std::string& lname, const fmap_shape& shape1, const fmap_shape& shape2, bool total){
	std::cerr << "The h*w of inputs of Layer " << lname << " mismatch!" << std::endl;
	if(total){
		std::cerr << "\t Real ifmap: " << shape1 << ", Total ifmap: " << shape2 << '.'<< std::endl;
	}else{
		std::cerr << "\tH*W: (" << shape1.h << ',' << shape1.w << ") with (" << shape2.h << ',' << shape2.w << ")."<< std::endl;
	}
	throw std::logic_error("Input dimension mismatch.");
}

void Network::err_eltwise(const std::string& lname, const len_t from_C, const len_t add_C, const len_t elt_C){
	std::cerr << "The channel of inputs of EltwiseLayer " << lname << " mismatch!" << std::endl;
	std::cerr << "\t C: (" << from_C << ", " << from_C + add_C << ") should not include multiples of " << elt_C << '.'<< std::endl;
	throw std::logic_error("Eltwise ifmap channel mismatch.");
}

lid_t Network::add(Layer* l, const layer_set& ifmPrevs, bwidth_t width, const std::vector<InputData>& ifmInputData, const layer_set& wgtPrevs,const std::vector<InputData>& wgtInputData){
	// If no prevs indicated, use default_bs.
	l->ifm_input_data = ifmInputData;
	l->wgt_input_data = wgtInputData;

	bool default_prev = (ifmInputData.empty() && ifmPrevs.empty());
	lid_t last_id = 0;
	std::vector<lid_t> prev_layers, prevWgts;
	if(default_prev){
		assert(!layers.empty());
		assert(wgtPrevs.empty());
		last_id = static_cast<lid_t>(layers.size()-1);
		prev_layers.push_back(last_id);
	}else{
		for(const lid_t& i: ifmPrevs) prev_layers.push_back(i);
		for(const lid_t& i: wgtPrevs) prevWgts.push_back(i);
	}

	// total ifmap shape is already been padded
	const auto& total_ifmap_shape = l->tot_ifmap_shape();
	const auto& ofm_shape = l->ofmap_shape();

	fmap_shape padded_ifm=ofm_shape;
	auto origin_ifm_size=total_ifmap_shape.size;
	padded_ifm.c = 0;

	auto check_func = [&](const fmap_shape& in_shape){
		// if(in_shape.size % (padded_ifm.h*padded_ifm.w) != 0)
		// {
		// 	DEBUG("in_shape",in_shape);
		// 	DEBUG("padded_ifm",padded_ifm);
		// 	DEBUG("layer_name",l->get_name());
		// 	DEBUG("in_shape.size",in_shape.size);
		// 	DEBUG("padded_ifm.h*padded_ifm.w",padded_ifm.h*padded_ifm.w);
		// 	DEBUG("in_shape.size % (padded_ifm.h*padded_ifm.w)",in_shape.size % (padded_ifm.h*padded_ifm.w));
		// }
		// assert(in_shape.size % (padded_ifm.h*padded_ifm.w) == 0);

		auto added_size=MIN(origin_ifm_size,in_shape.size);
		if(added_size%(padded_ifm.h*padded_ifm.w) != 0){
			DEBUG("in_shape",in_shape);
			DEBUG("padded_ifm",padded_ifm);
			DEBUG("layer_name",l->get_name());
			DEBUG("added_size",added_size);
			DEBUG("origin_ifm_size",origin_ifm_size);
			DEBUG("in_shape.size",in_shape.size);
			assert(added_size%(padded_ifm.h*padded_ifm.w) == 0);
		}
		auto added_c=added_size / (padded_ifm.h*padded_ifm.w);
		padded_ifm.c+=added_c;
	};

	// Check ext data fmap shapes
	for(const InputData& input : ifmInputData){
		const fmap_shape& in_shape = input.get_shape();
		check_func(in_shape);
	}
	// The number of external ifmap channels
	len_t external_C = padded_ifm.c;

	// Constraint for eltwise layer
	// seems not reasonable, beacuse not support for adding of two external data
	// if(eltwise_C > 0 && external_C > eltwise_C){
	// 	DEBUG("EltwiseLayer Error1",eltwise_C,external_C);
	// 	err_eltwise(l->get_name(), 0, external_C, eltwise_C);
	// }

	// Check prev layer fmap shapes
	vol_t prev_size=0;
	FOR_BITSET(it, prev_layers){
		lid_t layer_id = it;
		const fmap_shape& in_shape = getNode(layer_id).layer().ofmap_shape();
		prev_size += in_shape.size;
		// DEBUG("in_shape",in_shape);
		// DEBUG("name",getNode(layer_id).layer().get_name());
		// DEBUG("padded_ifm",padded_ifm);
		// DEBUG("padded_ifm",padded_ifm);
	}
	if(prev_size>0){
		//DEBUG("prev_size",prev_size);
		check_func(fmap_shape(prev_size, 1, 1));
	}

	// Check whether padding is valid
	padded_ifm.update_size();
	if(padded_ifm.size<origin_ifm_size){
		padded_ifm=total_ifmap_shape;
	}
	if(padded_ifm.size!=l->tot_ifmap_shape().size){
		DEBUG("padded_ifm",padded_ifm);
		DEBUG("l->tot_ifmap_shape()",l->tot_ifmap_shape());
		DEBUG("name",l->get_name());
		assert(padded_ifm.size==l->tot_ifmap_shape().size);
	}
	if(!const_cast<Layer*>(l)->set_padded_ifm(padded_ifm)){
		DEBUG("padded_ifm",padded_ifm);
		DEBUG("l->output_shape()",l->ofmap_shape());
		err_mismatch(l->get_name(), padded_ifm, l->tot_ifmap_shape(), true);
	}

	// Checks for weight prevs
	// will be delayed to calcNoC
	// if(prevWgts.count() > 0){
	// 	assert(l->weight_size() > 0);
	// 	fmap_shape wgtShape = l->weight_shape();
	// 	len_t curSize = 0;

	// 	FOR_BITSET(it, prevWgts){
	// 		lid_t layer_id = it;
	// 		const fmap_shape& out_shape=getNode(layer_id).layer().ofmap_shape();
	// 		//Notice: auto transposed ofmap shape from prev!
	// 		if(out_shape.size % wgtShape.h != 0)
	// 		{
	// 			DEBUG("out_shape",out_shape);
	// 			DEBUG("wgtShape",wgtShape);
	// 			DEBUG("layer_name",l->get_name());
	// 		}
	// 		assert(out_shape.size % wgtShape.h == 0);
	// 		auto added_size=MIN(out_shape.size,wgtShape.size);
	// 		curSize+=added_size;
	// 	}
	// 	//assert(curSize==wgtShape.size);
	// }
	if(layers.size() >= std::numeric_limits<lid_t>::max()){
		throw std::overflow_error("Too many layers! Consider using a larger format for lid_t (perhaps uint32_t?)");
	}

	// if(layers.size() >= prev_layers.size()){
	// 	throw std::overflow_error("Too many layers! Consider using a larger Bitset (perhaps 4096?)");
	// }

	lid_t cur_id = static_cast<lid_t>(layers.size());

	// Add next info of previous layers
	FOR_BITSET(it, prev_layers){
		lid_t layer_id = it;
		layers[layer_id].add_next(cur_id);
	}
	FOR_BITSET(it, prevWgts){
		lid_t layer_id = it;
		layers[layer_id].add_next(cur_id);
	}

	// Add layer to network
	layers.emplace_back(l, prev_layers, external_C, width, prevWgts);

	return cur_id;
}

void Network::reset()
{
	for(Node& n:layers){
		n.reset();
	}
}

Node& Network::getNode(lid_t id){
	Node& r=layers[id];
	return r;
}

const Node& Network::operator[](lid_t id) const{
	return layers[id];
}

lid_t Network::len() const{
	return static_cast<lid_t>(layers.size());
}

bool Network::is_chain() const{
	for(size_t i=1;i<layers.size();++i){
		const auto& prevs=layers[i].getPrevs();
		if(std::find(prevs.begin(),prevs.end(),i-1)==prevs.end()) return false;
	}
	return true;
}

bool Network::has_dep(const std::vector<lid_t>& src, const std::vector<lid_t>& dst) const{
	lid_t layer_id;
	if(src.size()>3){
		// O(C*d)
		std::vector<lid_t> b;
		FOR_BITSET(it, dst){
			layer_id = it;
			const auto& prevs=layers[layer_id].getPrevs();
			b.insert(b.end(), prevs.begin(), prevs.end());
		}
		FOR_BITSET(it, src){
			if(std::find(b.begin(),b.end(),it)!=b.end()) return true;
		}
	}else{
		// O(d*s)
		lid_t srcs[3];
		size_t src_num = 0;
		FOR_BITSET(it, src){
			srcs[src_num++]=it;
		}
		FOR_BITSET(it, dst){
			const auto& dst_set=layers[it].getPrevs();
			for(size_t i=0; i<src_num; ++i){
				if(std::find(dst_set.begin(),dst_set.end(),srcs[i])!=dst_set.end()) return true;
			}
		}
	}
	return false;
}

void Network::set_utime(const CoreMapper& mapper) const{
	for(const Node& n:layers){
		Layer& l = const_cast<Layer&>(n.layer());
		mapper.set_utime(l);
	}
}
