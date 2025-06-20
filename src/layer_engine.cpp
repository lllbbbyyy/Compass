#include "layer_engine.h"

#include <cassert>
#include <memory>

#include "network.h"
#include "util.h"
#include "debug.h"


// bool LayerScheme::isValid() const{
// 	return totCost.isValid();
// }

bool CompassLayerEngine::debug_detail = false;

LayerEngine::LayerCost::LayerCost(energy_t _energy, cycle_t _time)
	:energy(_energy),time(_time){}

LayerEngine::LayerCost&LayerEngine::LayerCost::operator+=(const LayerEngine::LayerCost& other){
	if(!(isValid() && other.isValid())){
		energy = energy_inf;
		return *this;
	}
	energy += other.energy;
	time += other.time;
	return *this;
}

LayerEngine::LayerCost& LayerEngine::LayerCost::operator*=(len_t other){
	if(isValid()){
		energy *= other;
		time *= other;
	}
	return *this;
}

bool LayerEngine::LayerCost::operator!=(const LayerEngine::LayerCost& other) const{
	if(isValid() && other.isValid())
		return energy != other.energy || time != other.time;
	return isValid() == other.isValid();
}

bool LayerEngine::LayerCost::isValid() const{
	return energy < energy_inf;
}

cost_t LayerEngine::LayerCost::cost(len_t nbatch) const{
	return energy*time*nbatch;
}

std::ostream& operator<<(std::ostream& os, const LayerEngine::LayerCost& cost){
	return os << "E:" << cost.energy << ", T:" << cost.time << ", Cost:" << cost.cost();
}


vol_t CompassLayerEngine::get_ubuf_size(const CoreMapper& mapper) const{
	return mapper.get_ubuf_size();
}

void CompassLayerEngine::calcNoC(vol_t Ifmfactor,vol_t Wgtfactor,vol_t Ofmfactor) const{
    noc->clear();

    const bool wgt_B = layerNode.hasWgtPrevs();
    const auto& nowLayer=layerNode.layer();

    // Fetch wgt input data
    vol_t wgtSizeInputData=0;
    for(const auto& inputData:nowLayer.wgt_input_data){
        auto inputSize=inputData.get_shape().tot_size(batchSize)*Wgtfactor;
        noc->unicast_from_dram(corePos, inputSize,inputData.DRAM_id);
        wgtSizeInputData+=inputSize;
    }

    auto weiShape=nowLayer.weight_size()*Wgtfactor-wgtSizeInputData;
    // TODO: need to process Whether weight comes from prev layer's fmap (e.g. in GroupConv)
    if(wgt_B){
        const auto& prevs = layerNode.getWgtPrevs();
        FOR_BITSET(it, prevs){
            const lid_t prev = it;
            const auto& prevNode=net->getNode(prev);
            const auto& prevLayer=prevNode.layer();
            auto ofmSize=prevLayer.ofmap_shape().tot_size(batchSize);
            vol_t needSize=MIN(weiShape,ofmSize)*Wgtfactor;

            if(runtimeInfo.layersWeiPrevDRAM[batchID][layerID].count(prev)){
                noc->unicast_from_dram(corePos, needSize,prevNode.writeDRAMIndex);
            }
            else{
                auto chipID=layerToChip[batchID][prev];
                if(prevNode.isTiling){
                    auto nocSize=MIN(weiShape,prevNode.htile*prevNode.ktile)*Wgtfactor;
                    noc->unicast({mlen_t(chipID%noc->xlen),mlen_t(chipID/noc->xlen)}, corePos, nocSize);
                    noc->unicast_from_dram(corePos, needSize-nocSize,prevNode.writeDRAMIndex);
                }
                else{
                    noc->unicast({mlen_t(chipID%noc->xlen),mlen_t(chipID/noc->xlen)}, corePos, needSize);
                }
            }
        }
    }
    else{
        if(isLoadWeight){
            noc->unicast_from_dram(corePos, weiShape,layerNode.readWgtDRAMIndex);
        }
    }

    //TODO: need to process the case of eltwise layer

    // Fetch input data
    for(const auto& inputData:nowLayer.ifm_input_data){
        auto inputSize=inputData.get_shape().tot_size(batchSize)*Ifmfactor;
        noc->unicast_from_dram(corePos, inputSize,inputData.DRAM_id);
    }


	// Fetch each prev layer from its ofmap/mem layout
    auto ifmSize=nowLayer.tot_ifmap_shape().tot_size(batchSize);

	const auto& prevs = layerNode.getIfmPrevs();
	FOR_BITSET(it, prevs){
        const lid_t prev = it;
        const auto& prevNode=net->getNode(prev);
        const auto& prevLayer=prevNode.layer();

        auto ofmSize=prevLayer.ofmap_shape().tot_size(batchSize);
        vol_t needSize=MIN(ifmSize,ofmSize)*Ifmfactor;


        if(runtimeInfo.layersIfmPrevDRAM[batchID][layerID].count(prev)||prevNode.mustReadDRAM){
            noc->unicast_from_dram(corePos, needSize,prevNode.writeDRAMIndex);
        }
        else{
            auto chipID=layerToChip[batchID][prev];
            if(prevNode.isTiling){
                auto nocSize=MIN(ifmSize,prevNode.htile*prevNode.ktile)*Ifmfactor;
                noc->unicast({mlen_t(chipID%noc->xlen),mlen_t(chipID/noc->xlen)}, corePos, nocSize);
                noc->unicast_from_dram(corePos, needSize-nocSize,prevNode.writeDRAMIndex);
            }
            else{
                noc->unicast({mlen_t(chipID%noc->xlen),mlen_t(chipID/noc->xlen)}, corePos, needSize);
            }
        }
	}

    auto ofmSize=nowLayer.ofmap_shape().tot_size(batchSize)*Ofmfactor;

    if(layerNode.isTiling){
        auto nocSize=layerNode.htile*layerNode.ktile*Ofmfactor;
        noc->unicast_to_dram(corePos, ofmSize-nocSize,layerNode.writeDRAMIndex);
        if(isWriteDram){
            noc->unicast_to_dram(corePos, nocSize,layerNode.writeDRAMIndex);
        }
    }
    else{
        if(isWriteDram){
            noc->unicast_to_dram(corePos, ofmSize,layerNode.writeDRAMIndex);
        }
    }
	// Save to remote mem if necessary
    return;
}

static std::tuple<len_t, len_t> tile_ptp_search(len_t m, len_t n, vol_t s) {
    assert(s >= 1);
    if(m*n<=s){
        return {m,n};
    }
    if(n<=s){
        return {s/n,n};
    }
    return {1,s};
}

static std::tuple<len_t, len_t, len_t> tile_search(len_t m, len_t k, len_t n, vol_t s) {
    assert(s >= 3);
    if(m*n+n*k+m*k<=s){
        return {m,k,n};
    }
    if(m+n+m*n<=s){
        return {m,(s-m*n)/(m+n),n};
    }
    if(1+2*n<=s){
        return {(s-n)/(1+n),1,n};
    }
    return {1,1,(s-1)/2};
}

CompassLayerEngine::LayerCost CompassLayerEngine::calCost(){
    LayerEngine::LayerCost cost;

    // Info extracted from layerNode
    auto& orilayer=layerNode.layer();
    const fmap_shape& ofmShape = orilayer.ofmap_shape();
    const Core::Buffer& ubuf = coreMapper->core().ubuf();
    const bool wgt_B = layerNode.hasWgtPrevs();

    // Estimate buffer usage. check overflow
    vol_t estimatedBuf = ofmShape.size;
    estimatedBuf+=orilayer.weight_size();
    estimatedBuf+=orilayer.real_ifmap_shape().tot_size(batchSize);
    std::shared_ptr<Layer> tilingLayer=nullptr;

    vol_t Ifmfactor=1, Wgtfactor=1,Ofmfactor=1;
    //DEBUG("layer",orilayer.get_name(),"estimatedBuf",estimatedBuf,"ubuf.Size",ubuf.Size,"batchSize",batchSize,"wgt_B",wgt_B);
    if(estimatedBuf > ubuf.Size){
        // tilingLayer=orilayer.clone();
        len_t mtile,ktile,ntile;
        
        if(IS_INSTANCE(&orilayer, ConvLayer)){
            const auto& wgtShape=orilayer.weight_shape();
            std::tie(mtile,ktile,ntile)=tile_search(ofmShape.h,wgtShape.h,wgtShape.c,ubuf.Size);
            //DEBUG("ofmShape.h",ofmShape.h,"wgtShape.h",wgtShape.h,"wgtShape.c",wgtShape.c,"ubuf.Size",ubuf.Size);
            //DEBUG("ktile",ktile,"mtile",mtile,"ntile",ntile);
            tilingLayer.reset(NLAYER(std::string("tilingLayer_")+orilayer.get_name(), Conv, C=ktile,K=ntile, H=mtile, W=1));           
            Ifmfactor=(ofmShape.c+ntile-1)/ntile;
            Wgtfactor=(ofmShape.h+mtile-1)/mtile;
            Ofmfactor=(wgtShape.h+ktile-1)/ktile;
            //DEBUG("Ifmfactor",Ifmfactor,"Wgtfactor",Wgtfactor,"Ofmfactor",Ofmfactor);
        }
        else if(IS_INSTANCE(&orilayer, EltwiseLayer)){
            // TODO: now only process EltwiseLayer is two source layer
            constexpr len_t nsrc=2;
            std::tie(mtile,ntile)=tile_ptp_search(ofmShape.h,ofmShape.c,ubuf.Size/(nsrc+1));
            ktile=0;
            tilingLayer.reset(NLAYER(std::string("tilingLayer_")+orilayer.get_name(), Eltwise, K=ntile, H=mtile, W=1,N=nsrc));
        }
        else if(IS_INSTANCE(&orilayer, PTPLayer)||IS_INSTANCE(&orilayer, TransposeLayer)){
            constexpr len_t nsrc=1;
            std::tie(mtile,ntile)=tile_ptp_search(ofmShape.h,ofmShape.c,ubuf.Size/(nsrc+1));
            ktile=0;
            tilingLayer.reset(NLAYER(std::string("tilingLayer_")+orilayer.get_name(), PTP, K=ntile, H=mtile, W=1));
        }
        else{
            DEBUG("layer",layerNode.layer().get_name(),"is not supported for tiling");
            assert(false);
        }
        tilingLayer->set_padded_ifm(tilingLayer->tot_ifmap_shape());
        // tilingLayer->isTiling=true;
        // tilingLayer->htile=mtile;
        // tilingLayer->ctile=ktile;
        // tilingLayer->ktile=ntile;
        layerNode.isTiling=true;
        layerNode.ctile=ktile;
        layerNode.ktile=ntile;
        layerNode.htile=mtile;
        // DEBUG("layer",layerNode.layer().get_name(),"estimatedBuf",estimatedBuf,"ubuf.Size",ubuf.Size);
        // DEBUG("layer",layerNode.layer().get_name(),"ofmShape",ofmShape,"weight_size",layer.weight_size(),"ifmap_shape",layer.real_ifmap_shape());
        // assert(estimatedBuf <= ubuf.Size); 
        // return {energy_inf,0};
    }
    // else{
    //     tilingLayer=nullptr;
    // }
    const auto& layer=tilingLayer?*tilingLayer:orilayer;

    //calc intra-chiplet dataflow
    auto mappingRes=coreMapper->genLayerMap(layer, batchSize, wgt_B);
    // if(!mappingRes.cost.is_valid()){
    //     DEBUG("layer",layer.get_name(),coreMapper->core().mac_num,coreMapper->core().getSRAMSize());
    // }
    assert(mappingRes.cost.is_valid());

    cost.energy = mappingRes.cost.energy*Ifmfactor*Wgtfactor*Ofmfactor;
    cost.time=mappingRes.cost.time*Ifmfactor*Wgtfactor*Ofmfactor;
    cost.calc_time=cost.time;
    cost.calc_energy=cost.energy;

    // Calculate ubuf energy
    const energy_t ubufOfm = ofmShape.tot_size(batchSize) * ubuf.RCost;
    const energy_t ubufWgt = isLoadWeight?layer.weight_size() * ubuf.WCost*Wgtfactor:0;
    const energy_t ubufIfm = layer.real_ifmap_shape().tot_size(batchSize) * ubuf.WCost*Ifmfactor;
	energy_t ubufTotal=ubufOfm+ubufWgt+ubufIfm;
    cost.energy+=ubufTotal;
    cost.ubuf_energy=ubufTotal;
    cost.ifm_ubuf_energy=ubufIfm;
    cost.wgt_ubuf_energy=ubufWgt;
    cost.ofm_ubuf_energy=ubufOfm;

    calcNoC(Ifmfactor,Wgtfactor);
    cycle_t nocTime = noc->get_time();
    if(debug_detail){
        DEBUG("name",layer.get_name());
        DEBUG("nocTime",nocTime);
        DEBUG("calcTime",cost.time);
    }
    cost.time = MAX(cost.time, nocTime);
    cost.noc_time = noc->get_hop_time();
    cost.dram_time = noc->get_DRAM_time();

    cost.energy += noc->get_cost();
    cost.dram_energy = noc->get_DRAM_cost();
    cost.noc_energy = noc->get_hop_cost();

    return cost;
}