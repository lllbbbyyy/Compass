#ifndef MODEL_ENGINE_H
#define MODEL_ENGINE_H

#include "util.h"
#include "network.h"
#include "layer_engine.h"
#include "core_mapping.h"
#include "assert.h"
#include "json.hpp"
#include <vector>


class ModelEngine{
public:
    virtual cycle_t calcLatency() = 0;
    virtual energy_t calcEnergy() = 0;
    virtual mc_t calcMonetaryCost() = 0;
    virtual std::pair<cycle_t,energy_t> calcLatencyAndEnergy() = 0;
};




class CompassModelEngine : public ModelEngine{
protected:
    std::shared_ptr<NoC> noc;
    std::vector<std::vector<cidx_t>> layerToChip;
    std::vector<int> segmentation;
    std::vector<std::vector<std::vector<cidx_t>>> segmentsChips;
    std::vector<std::vector<lid_t>> segmentsLayers;

    cidx_t coreNum;
    size_t batchDim;
    size_t segDim;
    size_t layerDim;
    std::vector<std::vector<cycle_t>> latencyRes;
public:
    struct MCCost_t{
        double compute_die_area;
        double IO_die_area;
        double os_area;
        double cost_compute;
        double cost_IO;
        double cost_os_overall;
        double cost_overall;
        friend std::ostream& operator<<(std::ostream& os, const MCCost_t& cost){
            os << "compute_die_area: " << cost.compute_die_area << std::endl;
            os << "IO_die_area: " << cost.IO_die_area << std::endl;
            os << "os_area: " << cost.os_area << std::endl;
            os << "cost_compute: " << cost.cost_compute << std::endl;
            os << "cost_IO: " << cost.cost_IO << std::endl;
            os << "cost_os_overall: " << cost.cost_os_overall << std::endl;
            os << "cost_overall: " << cost.cost_overall << std::endl;
            return os;
        }
    }mcCost;
    struct LatencyDetail_t{
        int batchID;
        lid_t layerID;
        cycle_t latencyBegin;
        cycle_t latencyEnd;

        cycle_t calcTime;
        cycle_t nocTime;
        cycle_t dramTime;

        friend std::ostream& operator<<(std::ostream& os, const LatencyDetail_t& latency){
            os<<"batchID:"<<latency.batchID<<" layerID:"<<latency.layerID<<" latencyBegin:"<<latency.latencyBegin<<" latencyEnd:"<<latency.latencyEnd<<std::endl;
            return os;
        }
    };
    std::vector<std::vector<LatencyDetail_t>> latencyDetail;
    struct EnergyDetail_t{
        int batchID;
        lid_t layerID;
        energy_t energy;

        energy_t calcEnergy;
        energy_t ubufEnergy;
        energy_t ifmUbufEnergy;
        energy_t wgtUbufEnergy;
        energy_t ofmUbufEnergy;
        energy_t nocEnergy;
        energy_t dramEnergy;
        
        friend std::ostream& operator<<(std::ostream& os, const EnergyDetail_t& energy){
            os<<"batchID:"<<energy.batchID<<" layerID:"<<energy.layerID<<" energy:"<<energy.energy<<std::endl;
            return os;
        }
    };
    std::vector<std::vector<EnergyDetail_t>> energyDetail;
    
    std::vector<std::shared_ptr<Network>> batchedModels;
    std::vector<std::shared_ptr<CoreMapper>> coreMappers;

    virtual cycle_t calcLatency() override;
    virtual energy_t calcEnergy() override;
    virtual mc_t calcMonetaryCost() override;
    virtual std::pair<cycle_t,energy_t> calcLatencyAndEnergy() override;

    CompassModelEngine(const std::vector<std::shared_ptr<Network>>& _batchedModels, const std::vector<std::shared_ptr<CoreMapper>>& _coreMappers, std::shared_ptr<NoC> _noc,const std::vector<int>& _segmentation, const std::vector<cidx_t>& _layerToChip) : coreMappers(_coreMappers){

        noc = std::make_shared<NoC>(*_noc);

        batchedModels.reserve(_batchedModels.size());
        for (const auto& ptr : _batchedModels) {
            batchedModels.push_back(std::make_shared<Network>(*ptr));  
        }

        coreNum = _coreMappers.size();
        batchDim = _batchedModels.size();
        assert(noc->xlen*noc->ylen==coreNum);
        std::vector<std::vector<cidx_t>> layerToChip;
        for(size_t i=0;i<batchedModels.size();i++){
            layerToChip.emplace_back(_layerToChip);
        }
        setSegmentation(_segmentation,layerToChip);
    };

    CompassModelEngine(const std::vector<std::shared_ptr<Network>>& _batchedModels, const std::vector<std::shared_ptr<CoreMapper>>& _coreMappers, std::shared_ptr<NoC> _noc,const std::vector<int>& _segmentation, const std::vector<std::vector<cidx_t>>& _layerToChip) : coreMappers(_coreMappers){

        noc = std::make_shared<NoC>(*_noc);

        batchedModels.reserve(_batchedModels.size());
        for (const auto& ptr : _batchedModels) {
            batchedModels.push_back(std::make_shared<Network>(*ptr));  
        }

        coreNum = _coreMappers.size();
        batchDim = _batchedModels.size();
        assert(noc->xlen*noc->ylen==coreNum);
        setSegmentation(_segmentation,_layerToChip);
    };

    CompassModelEngine(const CompassModelEngine& other):
        layerToChip(other.layerToChip),
        segmentation(other.segmentation),
        segmentsChips(other.segmentsChips),
        segmentsLayers(other.segmentsLayers),
        coreNum(other.coreNum),
        batchDim(other.batchDim),
        segDim(other.segDim),
        layerDim(other.layerDim),
        latencyRes(other.latencyRes),
        mcCost(other.mcCost),
        latencyDetail(other.latencyDetail),
        energyDetail(other.energyDetail),
        coreMappers(other.coreMappers)
    {
        noc = std::make_shared<NoC>(*other.noc);

        batchedModels.reserve(other.batchedModels.size());
        for (const auto& ptr : other.batchedModels) {
            batchedModels.push_back(std::make_shared<Network>(*ptr));  
        }
    };

    void setLayerToChip(const std::vector<std::vector<cidx_t>>& _layerToChip);

    void setSegmentation(const std::vector<int>& _segmentation, const std::vector<std::vector<cidx_t>>& _layerToChip);

    nlohmann::json get_latency_detail();

    nlohmann::json get_energy_detail();

    nlohmann::json get_mc_detail();
};

#endif // MODEL_ENGINE_H