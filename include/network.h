/* This file contains
 *	InputData: Describes the input data of the whole network.
 *  Node:      Represents a layer in the network.
 *  Network:   Represents a NN network.
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <cstdint>
#include <memory>
#include <vector>

#include "bitset.h"
#include "layer.h"
#include "util.h"

class CoreMapper;
//#include "coremapping.h"

/*
 * A node represents a layer in the network, with prev/next info.
 *
 * Here we use Node instead of directly using Layer, since
 * 1. There will be many kinds of derived class from layer (Conv, Pool, ...)
 *        where a single Node class is more friendly to Network.
 * 2. In the future one node may contain several layers,
 *        which means in this way we can maintain compatibility.
 */
class Node{
	// The underlying layer.
	std::shared_ptr<Layer> l;


public:
	bool isTiling=false;
	len_t ctile=1,ktile=1,htile=1;

	bool mustWriteDRAM=false;
	bool mustReadDRAM=false;
	int writeDRAMIndex=-1;
	int readWgtDRAMIndex=-1;
	
	void reset();

	Node(Layer* _l, const std::vector<lid_t>& _ifmPrevs, len_t _external_C, bwidth_t width = 0, const std::vector<lid_t>& _wgtPrevs = {});
	// Node(const Node& n) = delete;
	// Node(Node&& n)=default;

	// Getter functions.
	Layer& layer() const;
	const std::string& name() const;
	const std::vector<lid_t>& getIfmPrevs() const;
	const std::vector<lid_t>& getWgtPrevs() const;
	const std::vector<lid_t>& getPrevs() const;
	const std::vector<lid_t>& get_nexts() const;
	utime_t get_utime() const;
	len_t get_external_C() const;

	// Whether weight comes from prev layer's fmap (e.g. in GroupConv)
	bool hasWgtPrevs() const;

	// Adds l to "nexts"
	void add_next(lid_t l);

	~Node()=default;
};

class Network{
public:
	typedef std::vector<lid_t> layer_set;

private:
	std::vector<Node> layers;

	// Used to check data range validity.
	[[noreturn]] void err_mismatch(const std::string& lname, const fmap_shape& shape1, const fmap_shape& shape2, bool total=false);
	[[noreturn]] void err_eltwise(const std::string& lname, const len_t from_C, const len_t add_C, const len_t elt_C);

public:
	Network();
	// Network(const Network& n)=delete;
	// Network(Network&& n)=default;

	/*
	 * Append a new layer to the network
	 *
	 * [input]
	 *  l:        the layer to be added
	 *  ifmPrevs: previous layers (for ifmap)
	 *  width:    bitwidth (currently not used)
	 *  ext_data: external data (input of the network)
	 *  wgtPrevs: previous layers (for weight, used in GroupConv)
	 *
	 * [output]
	 *  index of the added layer
	 */
	lid_t add(Layer* l, const layer_set& ifmPrevs={}, bwidth_t width=0, const std::vector<InputData>& ifm_input_data={}, const layer_set& wgtPrevs={},const std::vector<InputData>& wgtInputData={});

	void reset();

	// Get the i-th node.
	Node& getNode(lid_t id);
	const Node& operator[](lid_t id) const;

	// Length of the network
	lid_t len() const;

	// Chain: a network where node i depends on node i-1
	bool is_chain() const;

	// Checks whether direct edge "s->d" exists for any s in src, d in dst.
	bool has_dep(const std::vector<lid_t>& src, const std::vector<lid_t>& dst) const;

	// Sets the utime of each node/layer. (utime: NPT in SET paper)
	void set_utime(const CoreMapper& mapper) const;

	~Network()=default;
};

#endif // NETWORK_H
