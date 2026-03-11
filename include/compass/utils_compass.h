#ifndef COMPASS_UTILS_COMPASS_H
#define COMPASS_UTILS_COMPASS_H

#include <vector>
#include <memory>
#include <utility>
#include <exception>
#include <cmath>
#include "util.h"
#include "core.h"
#include "noc.h"
#include "core_mapping.h"

// ------------------------------------------------------------------
inline constexpr int pex_num = 4;
inline constexpr int pey_num = 4;

// ------------------------------------------------------------------
std::vector<double> buffer_energy_table(int width, vol_t size);

std::pair<int, int> closest_factors(int n);

std::shared_ptr<CoreMapper>
createEyerissCoreMapper(int mac_num, vol_t ubufSize);

std::shared_ptr<CoreMapper>
createPolarCoreMapper(int mac_num, vol_t ubufSize,std::string mapping="",std::string macs="");

std::shared_ptr<NoC>
createNoC(mlen_t xlen, mlen_t ylen,
          bw_t noc_bw, bw_t dram_bw_each, int dram_num);

double cost_func(cycle_t latency, energy_t energy, mc_t mc);

std::pair<std::vector<int>,
          std::vector<std::vector<int>>>
random_mapping(size_t BATCH_SIZE,
               size_t LAYER_NUM,
               size_t CHIPLET_NUM);

#endif  // COMPASS_UTILS_COMPASS_H
