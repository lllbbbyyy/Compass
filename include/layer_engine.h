/* This file contains
 *	LayerScheme:    whole scheme of scheduling a layer
 *  LayerEngine:    base class for searching LayerScheme
 *  CompassLayerEngine: standard implementation of LayerEngine
 *
 *  One can add their own implementation of layer scheme searching as classes here.
 */

#ifndef LAYERENGINE_H
#define LAYERENGINE_H

#include <memory>
#include <set>
#include "core_mapping.h"
#include "noc.h"
#include "nns/nns.h"
#include "util.h"


// struct LayerScheme{
// 	// Total cost.
// 	SchNode::SchCost totCost;
// 	// Used to update external ubuf cost.
// 	energy_t extUbufEnergy;
// 	// Tiling scheme by CoreMapper.
// 	CoreMapper::CoreMapping tileSch;
// 	// Placement scheme
// 	PlaceSch place;
// 	// Noc Info
// 	NoC noc;

// 	// Whether scheme is valid (determined by totCost)
// 	bool isValid() const;
// };

class LayerEngine{
public:
    struct LayerCost{
        energy_t calc_energy;
        energy_t ubuf_energy;
        energy_t ifm_ubuf_energy;
        energy_t wgt_ubuf_energy;
        energy_t ofm_ubuf_energy;
        energy_t dram_energy;
        energy_t noc_energy;
		energy_t energy;
        cycle_t calc_time;
        cycle_t noc_time;
        cycle_t dram_time;
		cycle_t time;

		LayerCost(energy_t _energy=0, cycle_t _time=0);

		LayerCost& operator+=(const LayerCost& other);
		LayerCost& operator*=(len_t other);
		bool operator!=(const LayerCost& other) const;

		bool isValid() const;
		cost_t cost(len_t nbatch=1) const;

		friend std::ostream& operator<<(std::ostream& os, const LayerCost& cost);
    };
	virtual vol_t get_ubuf_size(const CoreMapper& coreMapper) const = 0;

	// Searches and returns best scheme for current layer
	//virtual LayerScheme search(LNode* curNode) const = 0;
    virtual LayerCost calCost()=0;
};

// for isWriteDRAM in modelEngine and for calcNoC in layerEngine
struct RuntimeInfo_t{
    // layerNextCnt[batchid][layerid]
    std::vector<std::vector<std::set<lid_t> > > layersIfmPrevDRAM;
    std::vector<std::vector<std::set<lid_t> > > layersWeiPrevDRAM;
    std::vector<std::vector<std::set<lid_t> > > layersNextDRAM;
    RuntimeInfo_t(const std::vector<std::shared_ptr<Network>>& models,size_t batchDim,size_t layerDim){
        layersIfmPrevDRAM.resize(batchDim,std::vector<std::set<lid_t>>(layerDim,std::set<lid_t>()));
        layersWeiPrevDRAM.resize(batchDim,std::vector<std::set<lid_t>>(layerDim,std::set<lid_t>()));
        layersNextDRAM.resize(batchDim,std::vector<std::set<lid_t>>(layerDim,std::set<lid_t>()));

        for(size_t i=0;i<batchDim;i++){
            auto& model=models[i];
            for(size_t j=0;j<layerDim;j++){
                auto& layer=model->getNode(j);
                auto& nextLayers=layer.get_nexts();
                FOR_BITSET(nextLayer,nextLayers){
                    layersNextDRAM[i][j].insert(nextLayer);
                }
                auto& prevIfmLayers=layer.getIfmPrevs();
                FOR_BITSET(nextLayer,prevIfmLayers){
                    layersIfmPrevDRAM[i][j].insert(nextLayer);
                }
                auto& prevWeiLayers=layer.getWgtPrevs();
                FOR_BITSET(nextLayer,prevWeiLayers){
                    layersWeiPrevDRAM[i][j].insert(nextLayer);
                }
            }
        }
    }
};

class CompassLayerEngine : public LayerEngine{

	// Calculates NoC *noc* from current placement *place*
	void calcNoC(vol_t Ifmfactor,vol_t Wgtfactor,vol_t Ofmfactor=1) const;

public:

    Node& layerNode;
    int batchID;
    lid_t layerID;
    std::shared_ptr<CoreMapper> coreMapper;
    std::shared_ptr<NoC> noc;
    std::shared_ptr<Network> net;
    pos_t corePos;
    //the segment list which the layer in
    const RuntimeInfo_t& runtimeInfo;
    const std::vector<std::vector<cidx_t>>& layerToChip;
    bool isWriteDram;
    bool isLoadWeight;

    std::unique_ptr<Layer> tilingLayer=nullptr;

    //useless
    len_t batchSize=1;

    static bool debug_detail;
    // CompassLayerEngine(const Node& _layerNode, const std::shared_ptr<CoreMapper> _coreMapper, std::shared_ptr<NoC> _noc,std::shared_ptr<Network> _net, pos_t _corePos, const std::vector<lid_t>& _segmentLayers,const std::vector<cidx_t>& _segmentCores,bool _isWriteDram=false,bool _isLoadWeight=false)
    //     :layerNode(_layerNode), coreMapper(_coreMapper), noc(_noc), net(_net), corePos(_corePos), segmentLayers(_segmentLayers),segmentCores(_segmentCores),isWriteDram(_isWriteDram),isLoadWeight(_isLoadWeight)
    // {
        
    // };


    CompassLayerEngine(Node& _layerNode,int _batchID,lid_t _layerID, const std::shared_ptr<CoreMapper> _coreMapper, std::shared_ptr<NoC> _noc,std::shared_ptr<Network> _net, pos_t _corePos, const RuntimeInfo_t& _runtimeInfo,const std::vector<std::vector<cidx_t>>& _layerToChip,bool _isWriteDram=false,bool _isLoadWeight=false)
        :layerNode(_layerNode),batchID(_batchID),layerID(_layerID) ,coreMapper(_coreMapper), noc(_noc), net(_net), corePos(_corePos), runtimeInfo(_runtimeInfo),layerToChip(_layerToChip),isWriteDram(_isWriteDram),isLoadWeight(_isLoadWeight)
    {
    };

	virtual vol_t get_ubuf_size(const CoreMapper& coreMapper) const override;
    virtual LayerCost calCost() override;
protected:
    CompassLayerEngine(const CompassLayerEngine& node) = default;
};

#endif // LAYERENGINE_H
