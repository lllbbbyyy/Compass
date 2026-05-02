/* This file contains
 *	pos_t:      Represents the position of a core
 *  fmap_shape: Represents the shape of fmap (C, H, W)
 *  fmap_range: Represents the range of a data block (C, B, H, W)
 *  Helper functions, type definitions, etc.
 */

#ifndef UTIL_H
#define UTIL_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <iostream>
#include <random>
#include <thread>
#include <mutex>
#include <functional>
#include <shared_mutex>
#include <map>
#include <cmath>
#include <optional>
#include <limits>
#include <type_traits>

#define KB *(int64_t)1024
#define MB *(int64_t)1024 KB

#define mm2 *(int64_t)1000000

// Need to guarantee that x is not 0
#define DIVCEIL(x,y) (((x)==0)?0:((x)-1)/(y)+1)
#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

#define IS_INSTANCE(ptr, cls) (dynamic_cast<const cls*>(ptr) != nullptr)
#define REF_IS_INSTANCE(ref, cls) IS_INSTANCE(&(ref), cls)

// Basic type definition.

#define DEF_MAX_(tname, max_name)\
constexpr tname max_name = (std::numeric_limits<tname>::has_infinity)?(std::numeric_limits<tname>::infinity()):(std::numeric_limits<tname>::max());

#define DEF_MAX(type) DEF_MAX_(type##_t, type##_inf)

typedef std::int64_t len_t;

// Volume of UBUF/REGF/... (in bytes)
typedef std::int64_t vol_t;

typedef vol_t hop_t;

typedef std::int64_t access_t;

typedef double energy_t;
DEF_MAX(energy);

typedef std::int64_t cycle_t;

typedef std::int32_t bw_t;
DEF_MAX(bw);

typedef double cost_t;
DEF_MAX(cost);

typedef double density_t;

typedef double mc_t;

typedef std::int32_t lid_t;

typedef std::int32_t cidx_t;

typedef std::int16_t mlen_t;
#define IO_INT8_

// The unit time of one layer (one core && one batch).
typedef double utime_t;

// The bitwidth of data (in bits).
typedef std::uint8_t bwidth_t;
#define IO_UINT8_

#undef DEF_MAX_
#undef DEF_MAX

#ifdef IO_INT8_
#undef IO_INT8_
std::istream& operator>>(std::istream& in, std::int8_t& num);
std::ostream& operator<<(std::ostream& out, const std::int8_t& num);
#endif

#ifdef IO_UINT8_
#undef IO_UINT8_
std::istream& operator>>(std::istream& in, std::uint8_t& num);
std::ostream& operator<<(std::ostream& out, const std::uint8_t& num);
#endif


// Function, struct and global variables definition.

extern vol_t ofm_ubuf_vol;

// part_intv guarantees that the first element is always non-zero.
extern len_t* part_intv(len_t tot_len, len_t ncuts);

struct pos_t{
	typedef std::uint64_t pos_hash_t;
	mlen_t x,y;

	bool operator<(const pos_t& other) const;
	bool operator==(const pos_t& other) const;
	bool operator>(const pos_t& other) const;
	bool operator<=(const pos_t& other) const;
	bool operator>=(const pos_t& other) const;
	bool operator!=(const pos_t& other) const;

	friend std::ostream& operator<<(std::ostream& os, const pos_t& pos);
};

struct pos_hash {
	std::size_t operator()(const pos_t& pos) const {
		return std::hash<pos_t::pos_hash_t>{}(*reinterpret_cast<const pos_t::pos_hash_t*>(&pos));
	}
};

struct fmap_shape{
	len_t c, h, w;
	vol_t size;

	fmap_shape()=default;
	fmap_shape(len_t _c, len_t _h, len_t _w=0);

	bool operator==(const fmap_shape& other) const;

	void update_size();
	vol_t tot_size(len_t batch_size) const;

	friend std::ostream& operator<<(std::ostream& os, const fmap_shape& shape);
};

/* the range is a left-closed & right-opened interval
 * i.e. [from.X, to.X)
 */
struct fmap_range{
	struct dim_range{
		len_t from, to;

		bool operator<(const dim_range& other) const;
		bool operator==(const dim_range& other) const;
		bool operator!=(const dim_range& other) const;
		dim_range& operator+=(const len_t& offset);
		dim_range& operator-=(const len_t& offset);

		bool is_empty() const;
		vol_t size() const;
		dim_range intersect(const dim_range& other) const;

		friend std::ostream& operator<<(std::ostream& os, const dim_range& range);
	}c, b, h, w;

	fmap_range()=default;
	fmap_range(const dim_range& _c, const dim_range& _b, const dim_range& _h, const dim_range& _w);
	explicit fmap_range(const fmap_shape& shape, len_t B = 1);

	bool operator<(const fmap_range& other) const;
	bool operator==(const fmap_range& other) const;

	vol_t size() const;
	bool is_empty() const;

	const dim_range& get_range(std::uint8_t idx) const;
	fmap_range intersect(const fmap_range& other) const;

	friend std::ostream& operator<<(std::ostream& os, const fmap_range& range);
};

class ThreadSafeRandom {
public:
    // Set global seed (optional)
    static void set_seed(unsigned int seed) {
        std::lock_guard<std::mutex> lock(seed_mutex);
        global_seed = seed;
    }

    // Get random integer in range [a, b]
    static int rand_int(int a, int b) {
        std::uniform_int_distribution<int> distribution(a, b);
        return distribution(get_generator());
    }

    static double rand_double(double a, double b) {
        std::uniform_real_distribution<double> distribution(a, b);
        return distribution(get_generator());
    }

    // Get random percentage in range [0.0, 1.0] (double)
    static double rand_percent() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        return distribution(get_generator());
    }

	static std::mt19937& get_generator() {
        thread_local std::mt19937 generator = [] {
            std::lock_guard<std::mutex> lock(seed_mutex);
            static unsigned int counter = 0;
            return std::mt19937(global_seed + counter++);
        }();
        return generator;
    }
private:
    static inline unsigned int global_seed = 42; // Default seed
    static inline std::mutex seed_mutex;
};

template <typename K, typename V>
class ThreadSafeMap {
public:
    ThreadSafeMap() = default;
    ~ThreadSafeMap() = default;

    ThreadSafeMap(const ThreadSafeMap&) = delete;
    ThreadSafeMap& operator=(const ThreadSafeMap&) = delete;


    void set(const K& key, const V& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        map_[key] = value;
    }


    std::optional<V> get(const K& key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void erase(const K& key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        map_.erase(key);
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.size();
    }

    bool empty() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.empty();
    }

    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        map_.clear();
    }

private:
    std::map<K, V> map_;
    mutable std::shared_mutex mutex_;
};

template<typename T>
struct NthRootResult {
    bool isInteger;      
    T nearestRoot;       
};

template<typename T>
T safePower(T base, int n) {
    static_assert(std::is_integral<T>::value, "T must be an integral type");
    
    if (n == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (base == -1) return (n % 2 == 0) ? 1 : -1;
    
    T result = 1;
    T absBase = (base < 0) ? -base : base;
    
    for (int i = 0; i < n; i++) {
        // check overflow
        if (result > std::numeric_limits<T>::max() / absBase) {
            return std::numeric_limits<T>::max(); 
        }
        result *= absBase;
    }
    
    if (base < 0 && n % 2 == 1) {
        result = -result;
    }
    
    return result;
}

template<typename T>
NthRootResult<T> nthRoot(T x, int n) {
    static_assert(std::is_integral<T>::value, "T must be an integral type");
    
    NthRootResult<T> result;
    result.isInteger = false;
    result.nearestRoot = 0;
    
    if (n <= 0) {
        return result;
    }
    
    if (n == 1) {
        result.isInteger = true;
        result.nearestRoot = x;
        return result;
    }
    
    if (x < 0 && n % 2 == 0) {
        return result;
    }
    
    if (x == 0) {
        result.isInteger = true;
        result.nearestRoot = 0;
        return result;
    }
    
    if (x == 1) {
        result.isInteger = true;
        result.nearestRoot = 1;
        return result;
    }
    
    if (x == -1 && n % 2 == 1) {
        result.isInteger = true;
        result.nearestRoot = -1;
        return result;
    }
    
    bool isNegative = (x < 0);
    T absX = isNegative ? -x : x;
    
    double estimate = pow(static_cast<double>(absX), 1.0 / n);
    T root = static_cast<T>(estimate);
    
    T power = safePower(root, n);
    
    while (root > 0 && power > absX) {
        root--;
        power = safePower(root, n);
    }
    
    while (power < absX) {
        T nextRoot = root + 1;
        T nextPower = safePower(nextRoot, n);
        
        if (nextPower == std::numeric_limits<T>::max() || nextPower < power) {
            break;
        }
        
        if (nextPower > absX) {
            break;
        }
        
        root = nextRoot;
        power = nextPower;
    }
    
    T lowerRoot = root;
    T lowerPower = safePower(lowerRoot, n);
    
    T upperRoot = root + 1;
    T upperPower = safePower(upperRoot, n);
    
    if (lowerPower == absX) {
        result.isInteger = true;
        result.nearestRoot = isNegative ? -lowerRoot : lowerRoot;
        return result;
    }
    
    if (upperPower == absX) {
        result.isInteger = true;
        result.nearestRoot = isNegative ? -upperRoot : upperRoot;
        return result;
    }
    
    // compare |absX - lowerPower| and |upperPower - absX|
    T lowerDiff = absX - lowerPower;
    T upperDiff = (upperPower == std::numeric_limits<T>::max()) ? std::numeric_limits<T>::max() : upperPower - absX;
    
    if (lowerDiff <= upperDiff) {
        result.nearestRoot = isNegative ? -lowerRoot : lowerRoot;
    } else {
        result.nearestRoot = isNegative ? -upperRoot : upperRoot;
    }
    
    return result;
}

#endif // UTIL_H
