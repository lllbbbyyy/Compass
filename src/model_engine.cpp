#include "model_engine.h"
#include "debug.h"
#include "json.hpp"
#include <memory>
#include <cmath>
#include <map>
#include <set>

cycle_t CompassModelEngine::calcLatency(){
    auto [allLatency,_]=calcLatencyAndEnergy();
    (void)_;
    return allLatency;
}


energy_t CompassModelEngine::calcEnergy(){
    auto [_,allEnergy]=calcLatencyAndEnergy();
    (void)_;
    return allEnergy;
}


std::pair<cycle_t,energy_t> CompassModelEngine::calcLatencyAndEnergy(){
    latencyDetail.clear();
    energyDetail.clear();
    latencyDetail.resize(coreNum);
    energyDetail.resize(coreNum);
    for(auto& model:batchedModels){
        model->reset();
    }
    /**determine isWriteDRAM**/
    std::vector<std::vector<bool > > isWriteDRAM(batchDim,std::vector<bool>(layerDim,true));
    
    auto runtimeInfo=RuntimeInfo_t(batchedModels,batchDim,layerDim);

    std::vector<std::pair<int,lid_t>> layerInfoPerChip(coreNum,{-1,0});
    for(size_t i = 0; i < segDim; i++){
        for(size_t j=0;j<batchDim;j++){
            for(size_t k=0;k<segmentsChips[j][i].size();k++)
            {
                auto chipID=segmentsChips[j][i][k];
                auto layerID=segmentsLayers[i][k];
                for(cidx_t l=0;l<coreNum;l++){
                    auto [preBatchID,preLayerID]=layerInfoPerChip[l];
                    if(preBatchID==int(j)){
                        runtimeInfo.layersNextDRAM[preBatchID][preLayerID].erase(layerID);
                        if(preBatchID!=-1&&runtimeInfo.layersNextDRAM[preBatchID][preLayerID].size()==0){
                            isWriteDRAM[preBatchID][preLayerID]=false;
                        }

                        runtimeInfo.layersIfmPrevDRAM[j][layerID].erase(preLayerID);
                        runtimeInfo.layersWeiPrevDRAM[j][layerID].erase(preLayerID);
                    }
                }

                layerInfoPerChip[chipID]={j,layerID};
            }
        }
    }
    /**end isWriteDRAM**/

    cycle_t allLatency=0;
    energy_t allEnergy=0;

    std::vector<cycle_t> latencyPerChip(coreNum,0);
    std::map<std::pair<int,lid_t>,cycle_t> layer2Latency;
    std::vector<lid_t> lastLayerPerChip(coreNum,0);

    for(size_t i = 0; i < segDim; i++){
        for(size_t j=0;j<batchDim;j++){
            auto& model=batchedModels[j];

            for(size_t k=0;k<segmentsChips[j][i].size();k++)
            {
                auto chipID=segmentsChips[j][i][k];
                auto layerID=segmentsLayers[i][k];
                bool isWriteDram = isWriteDRAM[j][layerID]||model->getNode(layerID).mustWriteDRAM;
                bool isLoadWeight = false;
                if(latencyPerChip[chipID]==0||lastLayerPerChip[chipID]!=layerID){
                    isLoadWeight=true;
                }

                //DEBUG("j,layerid,isWriteDram,isLoadWeight",j,layerID,isWriteDram,isLoadWeight);

                //calc layer latency and energy
                auto engine = std::make_shared<CompassLayerEngine>(model->getNode(layerID),j,layerID,coreMappers[chipID],noc,model,pos_t{mlen_t(chipID%noc->xlen),mlen_t(chipID/noc->xlen)},runtimeInfo,layerToChip,isWriteDram,isLoadWeight);

                auto cost=engine->calCost();
                auto layerLatency=cost.time;
                auto layerEnergy=cost.energy;
                //end calc

                auto latency=latencyPerChip[chipID];
                const auto& prevLayers=model->getNode(layerID).getPrevs();
                FOR_BITSET(it,prevLayers){
                    auto prevLayer=lid_t(it);
                    if(layer2Latency.count({j,prevLayer})){
                        latency=std::max(latency,layer2Latency[{j,prevLayer}]);
                    }
                }

                layerLatency+=latency;
                latencyPerChip[chipID]=layerLatency;
                layer2Latency[{j,layerID}]=layerLatency;
                lastLayerPerChip[chipID]=layerID;

                allEnergy+=layerEnergy;

                //detail
                latencyDetail[chipID].push_back({int(j),layerID,latency,layerLatency,cost.calc_time,cost.noc_time,cost.dram_time});
                energyDetail[chipID].push_back({int(j),layerID,layerEnergy,cost.calc_energy,cost.ubuf_energy,cost.ifm_ubuf_energy,cost.wgt_ubuf_energy,cost.ofm_ubuf_energy,cost.noc_energy,cost.dram_energy});
                //end detail
            }
        }
    }
    allLatency=*std::max_element(latencyPerChip.begin(),latencyPerChip.end());
    return {allLatency,allEnergy};
}


static double wafer_util(double area) {
	if (0 < area&&area <= 25 mm2) {
		return 0.865;
	}
	else if (area <= 50 mm2) {
		return 0.859;
	}
	else if (area <= 100 mm2) {
		return 0.849;
	}
	else if (area <= 200 mm2) {
		return 0.815;
	}
	else if (area <= 300 mm2) {
		return 0.806;
	}
	else if (area <= 400 mm2) {
		return 0.792;
	}
	else if (area <= 500 mm2) {
		return 0.778;
	}
	else if (area <= 600 mm2) {
		return 0.764;
	}
	else if (area <= 700 mm2) {
		return 0.743;
	}
	else if (area <= 800 mm2) {
		return 0.743;
	}
    else{
        return 0.743;
    }
}

mc_t CompassModelEngine::calcMonetaryCost() {

	//DENSITY
	static constexpr density_t SRAM_den = 0.0964 * 8;
	static constexpr density_t MAC_den = 57.9;//8-bit
	static constexpr density_t LR_mac_den = MAC_den * 8;
	static constexpr density_t NoP_len = 333; 
	static constexpr density_t NoP_wid = 800;
	static constexpr density_t NoC_den = 16781.312;
	static constexpr density_t DDR_PHY_den = 6.53 * 1000000;
	static constexpr density_t DDR_ctrl_den = 510 * 1000 * 0.09;
	double yield = 0.9;

	//COST
	static constexpr double cost_silicon_mm_compute = 0.084887 ;
	static constexpr double cost_silicon_mm_IO = 0.056383;
	static constexpr double cost_os = 0.005; //per mm^2
	static constexpr double os_area_scale_factor = 4;// os area will be larger than chip
	static constexpr double ddr_cost = 3.5;//per 16bithttps://www.dramexchange.com/
	double os_cost_factor;//larger os will be more expensive 
	static constexpr double post_layout_scale = 2.2;
	static constexpr double control_unit_prop = 1.05;
	static constexpr double DFT_prop = 1.05;
	//*********************DIE AREA*********************
    //auto compute_die_num=coreMappers.size();

    double compute_die_area=0;
    std::vector<double> compute_dir_areas;
    for(auto& coreMapper:coreMappers){
        auto& core=coreMapper->core();
        double sram_area_per_core = core.getSRAMSize() * SRAM_den;
	    double mac_area_per_core = core.mac_num * MAC_den + core.LR_mac_num*LR_mac_den;
	    double NoC_area_per_core = noc->NoC_bw * NoC_den;
        double core_area = (sram_area_per_core + mac_area_per_core + NoC_area_per_core)*post_layout_scale;
        // DEBUG("1", sram_area_per_core, mac_area_per_core, NoC_area_per_core, core_area);
        // double core_len = sqrt(core_area);
        double NoP_len_per_core = noc->NoC_bw / 4 * NoP_len;
        double D2D_area=4*NoP_len_per_core * NoP_wid;
        double die_area=core_area+D2D_area;
        die_area*= (control_unit_prop * DFT_prop);
        assert(die_area < 858 mm2);
        compute_die_area+=die_area;
        compute_dir_areas.push_back(die_area);
    }
    
    //128GB/s has 3mm2 area
	double PCIe_area = 3 mm2 * noc->DRAM_total_bw / 128;
    // *2 refer to 2 side
	double IO_die_area = noc->DRAM_total_bw / 44.0 * (DDR_PHY_den+DDR_ctrl_den)+ PCIe_area+(NoP_len*NoP_wid* noc->NoC_bw / 4 *noc->ylen*2);//neglect other IOs

	IO_die_area *= (control_unit_prop * DFT_prop);
	double total_die_area = compute_die_area + IO_die_area;
	double os_area = os_area_scale_factor * total_die_area;

    // DEBUG("compute_die_area,IO_die_area,os_area",compute_die_area/(1 mm2),IO_die_area/(1 mm2),os_area/(1 mm2));
	// Check if areas are too large.
	// if (compute_die_num == 1) {
	// 	if (total_die_area > 858 mm2) {
	// 		throw std::invalid_argument("`total_die_area` too large.");
	// 	}
	// }else{
	// 	if(IO_die_area>858 mm2 || compute_die_area>858 mm2) {
	// 		throw std::invalid_argument("`IO_die_area` or `compute_die_area` too large.") ;
	// 	}
	// }
	//*********************DIE AREA*********************
    if (os_area <= 30.0 * 30 mm2) {
		os_cost_factor = 1.5;
	}
	else if (os_area <= 55.0 * 55 mm2) {
		os_cost_factor = 2;
	}
	else {
		os_cost_factor = 4;
	}
	//*********************COST CALC*********************
	double cost_overall = 0;
	double yield_compute_die = 0;
	double yield_IO_die = 0;
	double cost_compute = 0;
	double cost_IO = 0;
	double cost_os_overall = 0;
	//double yield_soc = 0;
	//double cost_soc = 0;

    for(auto die_area:compute_dir_areas){
        yield_compute_die = pow(yield, die_area / (40 mm2));
        cost_compute += die_area / yield_compute_die / 1000000 * cost_silicon_mm_compute/wafer_util(die_area);
    }
    double IO_die_per_area=IO_die_area / noc->DRAM_num;
    yield_IO_die = pow(yield, IO_die_per_area / (40 mm2));
    cost_IO = IO_die_area / 1000000 * cost_silicon_mm_IO/ wafer_util(IO_die_per_area)/yield_IO_die + noc->DRAM_total_bw / 44.0 * ddr_cost;
    cost_os_overall = os_area * os_cost_factor / 1000000 * cost_os;
    cost_overall = cost_compute + cost_IO + cost_os_overall;

    mcCost.compute_die_area=compute_die_area;
    mcCost.IO_die_area=IO_die_area;
    mcCost.os_area=os_area;
    mcCost.cost_compute=cost_compute;
    mcCost.cost_IO=cost_IO;
    mcCost.cost_os_overall=cost_os_overall;
    mcCost.cost_overall=cost_overall;
    return cost_overall;
}

void CompassModelEngine::setLayerToChip(const std::vector<std::vector<cidx_t>>& _layerToChip){
    assert(_layerToChip.size() == batchDim);
    layerToChip.clear();
    layerToChip.reserve(batchDim);
    for(size_t j = 0; j < _layerToChip.size(); ++j){
        auto expanded = batchedModels[j]->expand_mapping_to_exec(_layerToChip[j]);
        assert(expanded.size() == layerDim);
        layerToChip.push_back(expanded);
    }
    segmentsChips.clear();
    segmentsChips.resize(batchDim);
    size_t j=0;
    for(auto& l2c:layerToChip){
        assert(l2c.size() == layerDim);
        std::vector<cidx_t> curChips;
        for(size_t i = 0; i < layerDim; i++){
            curChips.push_back(l2c[i]);
            if(i<segmentation.size() && segmentation[i]==1){
                segmentsChips[j].push_back(curChips);
                curChips.clear();
            }
        }
        if(!curChips.empty()){
            segmentsChips[j].push_back(curChips);
        }
        j+=1;
    }
}

void CompassModelEngine::setSegmentation(const std::vector<int>& _segmentation, const std::vector<std::vector<cidx_t>>& _layerToChip){

    layerDim = batchedModels[0]->len();
    segmentation = batchedModels[0]->expand_mapping_segmentation(_segmentation);
    assert(segmentation.size() == (layerDim == 0 ? 0 : layerDim - 1));
    segmentsLayers.clear();
    std::vector<lid_t> curLayers;
    for(size_t i = 0; i < layerDim; i++){
        curLayers.push_back(i);
        if(i<segmentation.size() && segmentation[i]==1){
            segmentsLayers.push_back(curLayers);
            curLayers.clear();
        }
    }
    if(!curLayers.empty()){
        segmentsLayers.push_back(curLayers);
    }
    segDim = segmentsLayers.size();
    setLayerToChip(_layerToChip);
}

nlohmann::json CompassModelEngine::get_latency_detail()
{
    nlohmann::json j;
    for (size_t i = 0; i < latencyDetail.size(); i++)
    {
        // cout<<"	core "<<i<<": "<<endl;
        j["core" + std::to_string(i)] = nlohmann::json::array();
        for (auto &detail : latencyDetail[i])
        {
            // cout<<"		"<<detail;
            nlohmann::json temp;
            temp["layerID"] = detail.layerID;
            temp["batchID"] = detail.batchID;
            temp["latencyBegin"] = detail.latencyBegin;
            temp["latencyEnd"] = detail.latencyEnd;
            temp["calcTime"]=detail.calcTime;
            temp["nocTime"]=detail.nocTime;
            temp["dramTime"]=detail.dramTime;
            auto& model = batchedModels[detail.batchID];
            auto mappingID = model->mapping_id_for_exec(detail.layerID);
            temp["layerName"]=model->getNode(detail.layerID).name();
            temp["mappingNodeID"]=mappingID;
            temp["mappingNodeName"]=model->getMappingNode(mappingID).name;
            j["core" + std::to_string(i)].push_back(temp);
        }
    }
    return j;
}

nlohmann::json CompassModelEngine::get_energy_detail()
{
    nlohmann::json j;
    for (size_t i = 0; i < energyDetail.size(); i++)
    {
        // cout<<"	core "<<i<<": "<<endl;
        j["core" + std::to_string(i)] = nlohmann::json::array();
        for (auto &detail : energyDetail[i])
        {
            // cout<<"		"<<detail;
            nlohmann::json temp;
            temp["layerID"] = detail.layerID;
            temp["batchID"] = detail.batchID;
            temp["energy"] = detail.energy;
            temp["calcEnergy"]=detail.calcEnergy;
            temp["ubufEnergy"]=detail.ubufEnergy;
            temp["nocEnergy"]=detail.nocEnergy;
            temp["dramEnergy"]=detail.dramEnergy;
            auto& model = batchedModels[detail.batchID];
            auto mappingID = model->mapping_id_for_exec(detail.layerID);
            temp["layerName"]=model->getNode(detail.layerID).name();
            temp["mappingNodeID"]=mappingID;
            temp["mappingNodeName"]=model->getMappingNode(mappingID).name;
            j["core" + std::to_string(i)].push_back(temp);
        }
    }
    return j;
}

nlohmann::json CompassModelEngine::get_mc_detail()
{
    nlohmann::json j;
    j["compute_die_area"] = mcCost.compute_die_area;
    j["IO_die_area"] = mcCost.IO_die_area;
    j["os_area"] = mcCost.os_area;
    j["cost_compute"] = mcCost.cost_compute;
    j["cost_IO"] = mcCost.cost_IO;
    j["cost_os_overall"] = mcCost.cost_os_overall;
    j["cost_overall"] = mcCost.cost_overall;
    return j;
}
