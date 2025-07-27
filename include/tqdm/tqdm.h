#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <cmath>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>

/*─────────────────────  tiny‑tqdm  ─────────────────────*/

class tqdm {
public:
    /*------------- Common usage: tqdm(N) -------------*/
    explicit tqdm(std::size_t total,
                std::string prefix = "",
                  std::size_t bar_width   = 50,
                  std::chrono::milliseconds min_update =
                      std::chrono::milliseconds(100))
        : total_(total),
          prefix_(std::move(prefix)),
          bar_width_(bar_width),
          min_update_(min_update),
          start_(clock::now()) {}

    /*------- Directly wrap any iterable object with begin/end -------*/
    template <class Range,
              class = decltype(std::begin(std::declval<const Range&>())),
              class = decltype(std::end  (std::declval<const Range&>()))>
    explicit tqdm(const Range& r,
                std::string prefix = "",
                  std::size_t bar_width   = 50,
                  std::chrono::milliseconds min_update =
                      std::chrono::milliseconds(100))
        : tqdm(static_cast<std::size_t>(std::distance(std::begin(r),
                                                      std::end(r))),std::move(prefix),
               bar_width, min_update) {}

    /*----------------- Iterator wrapper -----------------*/
    template <class UnderlyingIt>
    class iterator {
        using value_category_tag =
            typename std::iterator_traits<UnderlyingIt>::iterator_category;
    public:
        using iterator_category = value_category_tag;
        using value_type        = typename std::iterator_traits<UnderlyingIt>::value_type;
        using difference_type   = typename std::iterator_traits<UnderlyingIt>::difference_type;
        using pointer           = typename std::iterator_traits<UnderlyingIt>::pointer;
        using reference         = typename std::iterator_traits<UnderlyingIt>::reference;

        iterator(tqdm* owner, UnderlyingIt it) : owner_(owner), it_(it) {}
        reference operator*()  const { return *it_; }
        pointer   operator->() const { return std::addressof(*it_); }

        iterator& operator++() {
            ++it_;
            owner_->tick();
            return *this;
        }
        iterator operator++(int) { auto tmp=*this; ++(*this); return tmp; }

        bool operator!=(const iterator& rhs) const { return it_ != rhs.it_; }

    private:
        tqdm* owner_;
        UnderlyingIt it_;
    };

    /*------------ begin/end for `for(int i : tqdm(N))` ------------*/
    class int_iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const std::size_t*;
        using reference = const std::size_t&;

        int_iterator(tqdm* owner, std::size_t v) : owner_(owner), v_(v) {}
        std::size_t operator*() const { return v_; }

        int_iterator& operator++() {
            ++v_;
            owner_->tick();
            return *this;
        }
        bool operator!=(const int_iterator& rhs) const { return v_ != rhs.v_; }
    private:
        tqdm* owner_;
        std::size_t v_;
    };

    /*---------------- begin / end dual interface ----------------*/
    /* Integer iteration */
    int_iterator begin() { return int_iterator(this, 0); }
    int_iterator end()   { return int_iterator(this, total_); }

    /* Generic iteration: deduce underlying iterator type */
    template <class Range>
    auto wrap(Range& r) {
        using std::begin; using std::end;
        return std::make_pair(
            iterator(begin(r)), iterator(end(r)));
    }

    /*------------- Active manual update API --------------*/
    void tick(std::size_t n = 1) {
        current_ += n;
        auto now = clock::now();
        if (current_ == total_ || now - last_render_ >= min_update_)
            render(now);
    }

    ~tqdm() {  /* Ensure final 100% refresh and newline */
        render(clock::now(), true);
        std::cout << '\n';
    }

private:
    using clock = std::chrono::steady_clock;
    std::size_t total_;
    std::string prefix_;
    std::size_t bar_width_;
    std::chrono::milliseconds min_update_;
    clock::time_point start_;
    clock::time_point last_render_{start_};
    std::size_t current_ = 0;
    std::size_t last_units_drawn_ = 0;

    /*----------- Format time: hh:mm:ss -----------*/
    static std::string pretty_time(std::chrono::seconds s) {
        auto h = std::chrono::duration_cast<std::chrono::hours>(s);
        s -= std::chrono::duration_cast<std::chrono::seconds>(h);
        auto m = std::chrono::duration_cast<std::chrono::minutes>(s);
        s -= std::chrono::duration_cast<std::chrono::seconds>(m);
        std::ostringstream os;
        os << std::setfill('0') << std::setw(2) << h.count() << ':'
           << std::setw(2)      << m.count() << ':'
           << std::setw(2)      << s.count();
        return os.str();
    }

    void render(clock::time_point now, bool force=false) {
        float   progress = total_ ? float(current_) / total_ : 1.0f;
        std::size_t units_to_draw = static_cast<std::size_t>(std::round(progress * bar_width_));

        /* Skip if bar length unchanged and not forced (reduce flicker) */
        if (!force && units_to_draw == last_units_drawn_) return;
        last_units_drawn_ = units_to_draw;
        last_render_ = now;

        /* ETA / rate calculation */
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_);
        double rate = elapsed.count() ? current_ / double(elapsed.count()) : 0.0;
        double sec_left = rate ? (total_ - current_) / rate : 0.0;
        auto   eta      = std::chrono::seconds( static_cast<long long>(std::llround(sec_left)) );

        /* Assemble string */
        std::ostringstream os;
        os << "\33[2K\r";   // ← Erase line and return to start
        if (!prefix_.empty())
            os << prefix_ << " ";
        os << "[";
        for (std::size_t i = 0; i < bar_width_; ++i)
            os << (i < units_to_draw ? '=' : (i == units_to_draw ? '>' : ' '));
        os << "] "
           << std::setw(3) << int(progress * 100) << "% "
           << pretty_time(elapsed) << '<' << pretty_time(eta) << ", "
           << std::fixed << std::setprecision(3) << rate << " it/s";

        std::cout << os.str() << std::flush;
    }
};
