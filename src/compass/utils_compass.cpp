#include "compass/utils_compass.h"
#include "debug.h"
#include <cstdlib>   // rand, srand
#include <cassert>

// ------------------------------------------------------------
// buffer_energy_table 具体实现
std::vector<double> buffer_energy_table(int width, vol_t size) {
    std::vector<double> access(2);       // [0] = read, [1] = write
    vol_t size_;

    if (width >= 512) {
        size_  = size / (width / 256);
        access = buffer_energy_table(256, size_);
        access[0] *= std::sqrt(static_cast<double>(size) / size_);
        access[1] *= std::sqrt(static_cast<double>(size) / size_);
        return access;
    }

    if (width >= 256) {
        if (size >= 32 KB)       { access = {0.065678125, 0.0641375 }; }
        else if (size >= 16 KB)  { access = {0.06311875 , 0.05563125}; }
        else if (size >= 8 KB)   { access = {0.051290625, 0.04560625}; }
        else                     { access = {0.049     , 0.064     }; }
    }

    if (width >= 128) {
        if (size >= 32 KB)       { access = {0.106675   , 0.10539375}; }
        else if (size >= 16 KB)  { access = {0.06848125 , 0.06684375}; }
        else if (size >= 8 KB)   { access = {0.06553125 , 0.057975  }; }
        else if (size >= 4 KB)   { access = {0.05336875 , 0.0476625 }; }
        else                     { access = {0.06025625 , 0.06025625}; }
    }

    if (width >= 64) {
        if (size >= 32 KB)       { access = {0.194775 , 0.1923  }; }
        else if (size >= 16 KB)  { access = {0.112    , 0.110675}; }
        else if (size >= 8 KB)   { access = {0.0740875, 0.0722625}; }
        else if (size >= 4 KB)   { access = {0.0703625, 0.0626625}; }
        else                     { access = {0.057525 , 0.051775 }; }
    }

    if (width >= 32) {
        if (size >= 32 KB)       { access = {0.194775 , 0.1923 }; }
        else if (size >= 16 KB)  { access = {0.112    , 0.110675}; }
        else if (size >= 8 KB)   { access = {0.11608  , 0.13898 }; }
        else if (size >= 4 KB)   { access = {0.07422  , 0.09414 }; }
        else if (size >= 2 KB)   { access = {0.06326  , 0.08034 }; }
        else                     { access = {0.07952  , 0.0981  }; }
    }
    return access;
}

// ------------------------------------------------------------
// 其它函数实现
std::pair<int, int> closest_factors(int n) {
    int sqrt_n = static_cast<int>(std::sqrt(n));
    for (int i = sqrt_n; i >= 1; --i) {
        if (n % i == 0) {
            return {i, n / i};  // i ≤ n/i
        }
    }
    return {1, n};  // fallback（理论上不会走到这里）
}

std::shared_ptr<EyerissMapper>
createEyerissCoreMapper(int mac_num, vol_t ubufSize) {
    static constexpr double turnover_factor = 0.3 / 0.5;
    Core::numMac_t LR_mac_num = mac_num / (pex_num * pey_num);
    auto [arrayX, arrayY] = closest_factors(mac_num);

    energy_t LR_mac_cost = 0.0873;        // IEEE FP16
    EyerissCore::PESetting s2(arrayX, arrayY, 0.018);
    EyerissCore::Bus ibus(0.018, 64);
    EyerissCore::Bus wbus(0.018, 64);
    EyerissCore::Bus pbus(0.018, 64);
    EyerissCore::Buses   eBus{ ibus, wbus, pbus };
    EyerissCore::Buffers eBuf;

    eBuf.al1.Size = 32;
    eBuf.pl1.Size = 1;
    eBuf.wl1.Size = 128;
    eBuf.ul2.Size = ubufSize;

    eBuf.al1.RCost = 0.0509  * 8 * turnover_factor;
    eBuf.al1.WCost = 0.0506  * 8 * turnover_factor;
    eBuf.wl1.RCost = 0.0545  * 8 * turnover_factor;
    eBuf.wl1.WCost = 0.054   * 8 * turnover_factor;
    eBuf.pl1.RCost = eBuf.pl1.WCost = 0.0 * turnover_factor;
    eBuf.ul2.RCost = 0.1317125 * 8 * turnover_factor;
    eBuf.ul2.WCost = 0.234025  * 8 * turnover_factor;

    auto core = std::make_shared<EyerissCore>(s2, LR_mac_num, LR_mac_cost,
                                              eBus, eBuf);
    return std::make_shared<EyerissMapper>(core);
}

std::shared_ptr<PolarMapper>
createPolarCoreMapper(int mac_num, vol_t ubufSize) {
    auto [vector_len, lane_len] = closest_factors(mac_num);
    assert(vector_len % pex_num == 0);
    assert(lane_len   % pey_num == 0);
    vector_len /= pex_num;
    lane_len   /= pey_num;

    static constexpr double turnover_factor = 0.3 / 0.5;
    static constexpr energy_t LR_mac_cost   = 0.0873;

    Core::numMac_t LR_mac_num = mac_num / (pex_num * pey_num);

    PolarCore::Buffer al1, wl1, ol1, al2, wl2, ol2, ul3;
    PolarCore::PESetting s(vector_len, lane_len, 0.018);
    PolarCore::Bus bus(pex_num, pey_num, 0.018, 64);

    al1.Size = 8  * vector_len / 8 KB;
    ol1.Size = 2  * lane_len   / 8 KB;
    wl1.Size = 4  * lane_len * vector_len / 64 KB;
    ol2.Size = 28 * vector_len * lane_len / 64 KB;
    wl2.Size = 0;
    ul3.Size = ubufSize;

    al2.Size = 0;

    al1.RCost = (buffer_energy_table(vector_len * 8 , al1.Size)[0] +
                 0.1 * lane_len / 8) * 8 * turnover_factor;
    al1.WCost =  buffer_energy_table(vector_len * 8 , al1.Size)[1] *
                 8 * turnover_factor;
    wl1.RCost =  buffer_energy_table(vector_len * 8 , wl1.Size)[0] *
                 8 * turnover_factor;
    wl1.WCost =  buffer_energy_table(vector_len * 8 , wl1.Size)[1] *
                 8 * turnover_factor;
    ol1.RCost =  buffer_energy_table(lane_len * 16, ol1.Size)[0] *
                 8 * turnover_factor;
    ol1.WCost =  buffer_energy_table(lane_len * 16, ol1.Size)[1] *
                 8 * turnover_factor;
    ol2.RCost = 0.07648125 * 8 * ul3.Size / (1024 KB) * turnover_factor;
    ol2.WCost = 0.0989875 * 8 * ul3.Size / (1024 KB) * turnover_factor;
    ul3.RCost = 0.217125  * 8 * ul3.Size / (1024 KB) * turnover_factor;
    ul3.WCost = 0.234025  * 8 * ul3.Size / (1024 KB) * turnover_factor;
    al2.RCost = al2.WCost = 0;
    wl2.RCost = wl2.WCost = 0;

    auto core = std::make_shared<PolarCore>(
        s, LR_mac_num, LR_mac_cost, bus,
        PolarCore::Buffers{ al1, wl1, ol1, al2, wl2, ol2, ul3 });

    return std::make_shared<PolarMapper>(core);
}

std::shared_ptr<NoC>
createNoC(mlen_t xlen, mlen_t ylen,
          bw_t noc_bw, bw_t dram_bw_each, int dram_num) {

    static constexpr double hop_cost     = 2 * 8;
    static constexpr double DRAM_acc_cost = 10.5 * 8;

    std::vector<bw_t> dram_bws(dram_num, dram_bw_each);
    std::vector<pos_t> dram_router_list;
    std::vector<std::vector<pos_t>> dram_list;

    int dram_num_right=dram_num / 2;
    int dram_num_left=dram_num - dram_num_right;
    size_t left_router_num_per_dram = dram_num_left==0?0:ylen / dram_num_left;
    size_t right_router_num_per_dram = dram_num_right==0?0:ylen / dram_num_right;
    assert(left_router_num_per_dram > 0||right_router_num_per_dram);
    std::vector<pos_t> routers;

    // 左侧 DRAM
    if(left_router_num_per_dram>0){
        for (mlen_t y = 0; y < ylen; ++y) {
            dram_router_list.push_back({0, y});
            if (y % left_router_num_per_dram == 0 && !routers.empty()) {
                dram_list.push_back(routers);
                routers.clear();
            }
            routers.push_back({0, y});
        }
        
        if (!routers.empty()) {
            dram_list.push_back(routers);
            routers.clear();
        }
    }

    // 右侧 DRAM
    if(right_router_num_per_dram>0){
        for (mlen_t y = 0; y < ylen; ++y) {
            dram_router_list.push_back({static_cast<mlen_t>(xlen - 1), y});
            if (y % right_router_num_per_dram == 0 && !routers.empty()) {
                dram_list.push_back(routers);
                routers.clear();
            }
            routers.push_back({static_cast<mlen_t>(xlen - 1), y});
        }
        if (!routers.empty()) {
            dram_list.push_back(routers);
        }
    }

    return std::make_shared<NoC>(xlen, ylen,
                                 hop_cost, DRAM_acc_cost, noc_bw,
                                 dram_bws, dram_router_list, dram_list);
}

double cost_func(cycle_t latency, energy_t energy, mc_t mc) {
    return latency * energy * mc;
}

std::pair<std::vector<int>,
          std::vector<std::vector<int>>>
random_mapping(size_t BATCH_SIZE,
               size_t LAYER_NUM,
               size_t CHIPLET_NUM) {

    std::vector<int> segmentation;
    std::vector<std::vector<cidx_t>> layerToChip;

    // 随机分段
    for (size_t i = 0; i < LAYER_NUM - 1; ++i)
        segmentation.push_back(ThreadSafeRandom::rand_int(0,1));

    // 随机 layer→chiplet 映射
    for (size_t j = 0; j < BATCH_SIZE; ++j) {
        layerToChip.emplace_back();
        for (size_t i = 0; i < LAYER_NUM; ++i)
            layerToChip.back().push_back(ThreadSafeRandom::rand_int(0,CHIPLET_NUM-1));
    }
    return { segmentation, layerToChip };
}
