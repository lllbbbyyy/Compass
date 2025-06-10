#include "noc.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "util.h"
#include "debug.h"

NoC NoC::operator+(const NoC& other) const{
	NoC x = *this;
	return x += other;
}

NoC& NoC::operator+=(const NoC& other){
	tot_hops += other.tot_hops;
	tot_DRAM_acc += other.tot_DRAM_acc;
	if(calc_bw || other.calc_bw){
		assert(calc_bw && other.calc_bw);
		link_hops += other.link_hops;
	}
	return *this;
}

NoC NoC::operator*(const len_t& batch) const{
	NoC x = *this;
	return x *= batch;
}

NoC& NoC::operator*=(const len_t& batch){
	tot_hops *= batch;
	tot_DRAM_acc *= batch;
	if(calc_bw) link_hops *= batch;
	return *this;
}

NoC& NoC::operator/=(const len_t& batch){
	tot_hops /= batch;
	tot_DRAM_acc /= batch;
	if(calc_bw) link_hops /= batch;
	return *this;
}

void NoC::div(len_t batch){
	tot_hops /= batch;
	tot_DRAM_acc /= batch;
	if(calc_bw) link_hops.div(batch);
}

void NoC::clear(){
	tot_hops = 0;
	tot_DRAM_acc = 0;
	link_hops.clear();
	DRAM_access.clear();
	DRAM_access.resize(DRAM_num, 0);
}

access_t NoC::get_DRAM_acc(mlen_t i) const {
	//cycle_t dram_time = DIVCEIL(tot_DRAM_acc, (4*DRAM_bw));
	
	return DRAM_access[i];
}

cycle_t NoC::get_DRAM_time() const{
	cycle_t dram_time = 0;
	for(size_t i=0;i<DRAM_num;++i){
		dram_time = MAX(dram_time, DIVCEIL(DRAM_access[i], DRAM_bws[i]));
	}
	return dram_time;
}

cycle_t NoC::get_hop_time() const{
	return DIVCEIL(link_hops.max(), NoC_bw);
}

cycle_t NoC::get_time() const{
	cycle_t dram_time = get_DRAM_time();
	cycle_t noc_time = get_hop_time();
	return MAX(dram_time, noc_time);
}

energy_t NoC::get_cost() const{
	return get_hop_cost() + get_DRAM_cost();
}

energy_t NoC::get_hop_cost() const{
	return tot_hops * hop_cost;
}

energy_t NoC::get_DRAM_cost() const{
	return tot_DRAM_acc * DRAM_acc_cost;
}

hop_t NoC::get_tot_hops() const{
	return tot_hops;
}

access_t NoC::get_tot_DRAM_acc() const{
	return tot_DRAM_acc;
}

hop_t NoC::get_max_link() const{
	return link_hops.max();
}

// void NoC::fromRemoteMem(const DataLayout& toLayout){
// 	auto rLen = toLayout.rangeLength();
// 	for(cidx_t i=0; i<rLen; ++i){
// 		auto it = toLayout.at(i);
// 		vol_t curSize = it.range.size();
// 		if(curSize <= 0) continue;
// 		if(it.numTile == 1){
// 			unicast_from_dram(it.tiles[0], curSize);
// 		}else{
// 			multicast_from_dram(it.tiles, it.numTile, curSize);
// 		}
// 	}
// }

// void NoC::fromRemoteMem(const DataLayout& toLayout, len_t fromC, len_t toC){
// 	if(toC <= fromC) return;
// 	fmap_range::dim_range truncRange = {fromC, toC};

// 	auto rLen = toLayout.rangeLength();
// 	for(cidx_t i=0; i<rLen; ++i){
// 		auto it = toLayout.at(i);
// 		fmap_range range = it.range;
// 		range.c = range.c.intersect(truncRange);
// 		vol_t curSize = range.size();
// 		if(curSize <= 0) continue;
// 		if(it.numTile == 1){
// 			unicast_from_dram(it.tiles[0], curSize);
// 		}else{
// 			multicast_from_dram(it.tiles, it.numTile, curSize);
// 		}
// 	}
// }

// void NoC::toRemoteMem(const UniqueLayout& fromLayout){
// 	for(cidx_t i=0; i<fromLayout.totLength(); ++i){
// 		auto it = fromLayout[i];
// 		vol_t curSize = it.range.size();
// 		if(curSize <= 0) continue;
// 		unicast_to_dram(it.tile, curSize);
// 	}
// }

// void NoC::betweenLayout(const UniqueLayout& fromLayout, const DataLayout& toLayout, len_t fromCOffset, len_t fromB, len_t toB){
// 	hop_t h = 0;

// 	const auto* fLayout = dynamic_cast<const StdULayout*>(&fromLayout);
// 	if(fLayout == nullptr){
// 		// Currently only StdULayout implemented get_intersect()
// 		throw std::invalid_argument("betweenLayout() only implemented for StdULayout");
// 	}

// 	bool diffB = (fromB != toB);
// 	auto rLen = toLayout.rangeLength();

// 	for(cidx_t i=0; i<rLen; ++i){
// 		auto toEntry = toLayout.at(i);
// 		fmap_range toRange = toEntry.range;

// 		if(toRange.c.to <= fromCOffset) continue;
// 		toRange.c -= fromCOffset;

// 		for(auto it = fLayout->get_intersect(toRange, diffB); it.isValid(); it.next()){
// 			auto fromEntry = *it;
// 			vol_t v = calc_intersect(fromEntry.range, toRange, fromB, toB);
// 			if(v == 0) continue;

// 			if(toEntry.numTile == 1){
// 				h += unicastCalc(fromEntry.tile, *toEntry.tiles, v);
// 			}else{
// 				h += multicastCalc(fromEntry.tile, toEntry.tiles, toEntry.numTile, v);
// 			}
// 		}
// 	}

// 	tot_hops += h;
// }

std::ostream& operator<<(std::ostream& os, const NoC& noc){
	return os << "NoC(hops=" << noc.tot_hops << ", DRAM acc=" << noc.tot_DRAM_acc << ")";
}

std::vector<NoC::link_info> NoC::get_link_info() const{
	std::vector<link_info> info;
	if(!calc_bw) return info;

	info.reserve(link_hops.link_hops.size());
	for(const auto& it: link_hops.link_hops){
		mlen_t x, y, dir;
		HopCount::get_dir(it.first,xlen,ylen, x, y, dir);

		pos_t to;
		switch(dir){
		case 0:
			to = {static_cast<mlen_t>(x+1), y};
			break;
		case 1:
			to = {x, static_cast<mlen_t>(y-1)};
			break;
		case 2:
			to = {static_cast<mlen_t>(x-1), y};
			break;
		case 3:
			to = {x, static_cast<mlen_t>(y+1)};
			break;
		default:
			assert(false);
		}

		info.push_back({{x, y}, to, it.second * link_hops.factor});
	}

	// Sort in descending order.
	std::sort(info.rbegin(), info.rend());

	return info;
}

void NoC::unicast(pos_t src, pos_t dst, vol_t size){
	tot_hops += unicastCalc(src, dst, size);
}

hop_t NoC::unicastCalc(pos_t src, pos_t dst, vol_t size){
	assert(size >= 0);
	link_hops.flat_factor();
	if(calc_bw){
		size_t x_dir = (dst.x > src.x)?0:2;
		size_t y_dir = (dst.y > src.y)?3:1;
		mlen_t dx = (dst.x > src.x)?1:-1;
		mlen_t dy = (dst.y > src.y)?1:-1;
		for(mlen_t x = src.x; x != dst.x; x+= dx){
			link_hops.get(x, src.y, x_dir) += size;
		}
		for(mlen_t y = src.y; y != dst.y; y+= dy){
			link_hops.get(dst.x, y, y_dir) += size;
		}
	}
	return static_cast<hop_t>(abs(src.x-dst.x)+abs(src.y-dst.y)) * size;
}

void NoC::multicast(pos_t src, const pos_t* dst, cidx_t len, vol_t size){
	tot_hops += multicastCalc(src, dst, len, size);
}

hop_t NoC::multicastCalc(pos_t src, const pos_t* dst, cidx_t len, vol_t size){
	assert(size >= 0);
	link_hops.flat_factor();

	mlen_t cur_x = dst[0].x;
	mlen_t min_y = dst[0].y;
	hop_t h = 0;
	if(calc_bw){
		for(mlen_t x = src.x; x > dst[0].x; --x){
			link_hops.get(x, src.y, 2) += size;
		}
		for(mlen_t x = src.x; x < dst[len-1].x; ++x){
			link_hops.get(x, src.y, 0) += size;
		}
	}
	h += MAX(src.x, dst[len-1].x) - MIN(src.x, dst[0].x);

	for(cidx_t i=1; i<=len; ++i){
		if(i<len && dst[i].x == cur_x) continue;
		if(calc_bw){
			for(mlen_t y = src.y; y > min_y; --y){
				link_hops.get(cur_x, y, 1) += size;
			}
			for(mlen_t y = src.y; y < dst[i-1].y; ++y){
				link_hops.get(cur_x, y, 3) += size;
			}
		}
		h += MAX(src.y, dst[i-1].y) - MIN(src.y, min_y);
		if(i == len) break;
		cur_x = dst[i].x;
		min_y = dst[i].y;
	}
	return h * size;
}

void NoC::unicast_from_dram(pos_t dst, vol_t size){
	assert(size >= 0);
	size = size / DRAM_num;
	for (size_t m = 0; m < DRAM_num; m++) {
		auto llen = DRAM_list[m].size();
		size_t i = 0;
		vol_t from_size = 0;
		for (const pos_t& dram : DRAM_list[m]) {
			vol_t to_size = (size * ++i) / llen;
			unicast(dram, dst, to_size - from_size);
			from_size = to_size;
		}
		tot_DRAM_acc += size;
		DRAM_access[m] += size;
	}
}

void NoC::unicast_from_dram(pos_t dst, vol_t size,mlen_t dram_id){
	if(dram_id==-1){
		unicast_from_dram(dst, size);
		return;
	}
	auto llen = DRAM_list[dram_id].size();
	auto i = 0;
	vol_t from_size = 0;
	for(const pos_t& dram: DRAM_list[dram_id]) {
		vol_t to_size = (size * ++i) / llen;
		unicast(dram, dst, to_size - from_size);
		from_size = to_size;
	}
	tot_DRAM_acc += size;
	DRAM_access[dram_id] += size;
}

void NoC::unicast_to_dram(pos_t src, vol_t size){
	assert(size >= 0);
	size = size / DRAM_num;
	for (size_t m = 0; m < DRAM_num; m++) {
		auto llen = DRAM_list[m].size();
		size_t i = 0;
		vol_t from_size = 0;
		for (const pos_t& dram : DRAM_list[m]) {
			vol_t to_size = (size * ++i) / llen;
			unicast(src, dram, to_size - from_size);
			from_size = to_size;
		}
		tot_DRAM_acc += size;
		DRAM_access[m] += size;
	}
}

void NoC::unicast_to_dram(pos_t src, vol_t size,mlen_t dram_id){
	if(dram_id==-1){
		unicast_to_dram(src, size);
		return;
	}
	auto llen = DRAM_list[dram_id].size();
	size_t i = 0;
	vol_t from_size = 0;
	for(const pos_t& dram: DRAM_list[dram_id]) {
		vol_t to_size = (size * ++i) / llen;
		unicast(src, dram, to_size - from_size);
		from_size = to_size;
	}
	tot_DRAM_acc += size;
	DRAM_access[dram_id] += size;
}

void NoC::multicast_from_dram(const pos_t* dst, cidx_t len, vol_t size){
	assert(size >= 0);
	size = size / DRAM_num;
	for (size_t m = 0; m < DRAM_num; m++) {
		auto llen = DRAM_list[m].size();
		size_t i = 0;
		vol_t from_size = 0;
		for (const pos_t& dram : DRAM_list[m]) {
			vol_t to_size = (size * ++i) / llen;
			multicast(dram, dst, len, to_size - from_size);
			from_size = to_size;
		}
		tot_DRAM_acc += size;
		DRAM_access[m] += size;
	}
}

void NoC::multicast_from_dram(const pos_t* dst, cidx_t len, vol_t size,mlen_t dram_id){
	assert(size >= 0);
	auto llen = DRAM_list[dram_id].size();
	size_t i = 0;
	vol_t from_size = 0;
	for(const pos_t& dram: DRAM_list[dram_id]) {
		vol_t to_size = (size * ++i) / llen;
		multicast(dram, dst, len, to_size - from_size);
		from_size = to_size;
	}
	tot_DRAM_acc += size;
	DRAM_access[dram_id] += size;
}

vol_t NoC::calc_intersect(const fmap_range& rng1, const fmap_range& rng2, len_t bat1, len_t bat2){
	fmap_range ints = rng1.intersect(rng2);
	if(bat1 == bat2) return ints.size();

	len_t sb_st, sb_ed, lb_st, lb_ed, tot_b=0;
	if(bat1 > bat2){
		assert(bat1 % bat2 == 0);
		sb_st = rng2.b.from;
		sb_ed = rng2.b.to;
		lb_st = rng1.b.from;
		lb_ed = rng1.b.to;
		for(;sb_st < lb_ed; sb_st+=bat2, sb_ed+=bat2){
			if(sb_ed <= lb_st) continue;
			tot_b += MIN(sb_ed, lb_ed) - MAX(sb_st, lb_st);
		}
	}else{
		assert(bat2 % bat1 == 0);
		sb_st = rng1.b.from;
		sb_ed = rng1.b.to;
		lb_st = rng2.b.from;
		lb_ed = rng2.b.to;
		for(;sb_st < lb_ed; sb_st+=bat1, sb_ed+=bat1){
			if(sb_ed <= lb_st) continue;
			tot_b += MIN(sb_ed, lb_ed) - MAX(sb_st, lb_st);
		}
	}
	ints.b.from=0;
	ints.b.to=tot_b;
	vol_t v = ints.size();

	// If bat1 > bat2, reduce to bat2 batches.
	if(bat1 > bat2)  v /= (bat1 / bat2);
	return v;
}


HopCount& HopCount::operator+=(const HopCount& other){
	flat_factor();
	for(const auto& it : other.link_hops){
		link_hops[it.first] += it.second * other.factor;
	}
	return *this;
}

HopCount& HopCount::operator*=(const len_t& batch){
	factor *= batch;
	return *this;
}

HopCount& HopCount::operator/=(const len_t& batch){
	if(factor % batch == 0){
		factor /= batch;
	}else if(factor == 1){
		for(const auto& it: link_hops){
			assert(it.second % batch == 0);
			link_hops[it.first] = it.second / batch;
		}
	}else{
		for(const auto& it: link_hops){
			auto val = it.second * factor;
			assert(val % batch == 0);
			link_hops[it.first] = val / batch;
		}
		factor = 1;
	}
	return *this;
}

void HopCount::div(len_t batch){
	if(factor % batch == 0){
		factor /= batch;
	}else if(factor == 1){
		for(const auto& it: link_hops){
			// assert(it.second % batch == 0);
			link_hops[it.first] = it.second / batch;
		}
	}else{
		for(const auto& it: link_hops){
			auto val = it.second * factor;
			// assert(val % batch == 0);
			link_hops[it.first] = val / batch;
		}
		factor = 1;
	}
}

hop_t HopCount::max() const{
	hop_t h = 0;
	for(const auto& it: link_hops){
		h = MAX(h, it.second);
	}
	return h * factor;
}

hop_t& HopCount::get(mlen_t x, mlen_t y, mlen_t dir){
	assert(factor == 1);
	linkIdx_t idx = get_idx(x, y, dir);
	return link_hops[idx];
}

HopCount::linkIdx_t HopCount::get_idx(mlen_t x, mlen_t y, mlen_t dir){
	static_assert(sizeof(linkIdx_t) > 2 * sizeof(mlen_t), "linkIdx_t needs to store x, y and dir");

	linkIdx_t idx = (static_cast<linkIdx_t>(x) * ylen + y) * 4 + dir;
	assert(idx < static_cast<linkIdx_t>(4)*xlen*ylen);
	return idx;
}

void HopCount::get_dir(linkIdx_t link_idx,mlen_t xlen,mlen_t ylen, mlen_t& x, mlen_t& y, mlen_t& dir){
	// Notice: need to deal with negative x and y.
	(void)xlen;
	dir = link_idx % 4;
	link_idx /= 4;
	if(dir < 0){
		link_idx -= 1;
		dir += 4;
	}

	y = link_idx % ylen;
	link_idx /= ylen;
	if(y < 0){
		link_idx -= 1;
		y += ylen;
	}

	x = link_idx;
}

void HopCount::clear(){
	factor = 1;
	link_hops.clear();
}

void HopCount::flat_factor(){
	if(factor > 1){
		for(const auto& it: link_hops){
			link_hops[it.first] *= factor;
		}
		factor = 1;
	}
}


bool NoC::link_info::operator<(const link_info& other) const{
	if(total_hops != other.total_hops) return total_hops < other.total_hops;
	if(from != other.from) return from < other.from;
	return to < other.to;
}

bool NoC::link_info::operator==(const link_info& other) const{
	return total_hops == other.total_hops && from == other.from && to == other.to;
}

bool NoC::link_info::operator>(const link_info& other) const{
	if(total_hops != other.total_hops) return total_hops > other.total_hops;
	if(from != other.from) return from > other.from;
	return to > other.to;
}

std::ostream& operator<<(std::ostream& os, const NoC::link_info& info){
	return os<<info.from<<" -> "<<info.to<<'\t'<<info.total_hops;
}
