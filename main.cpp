/**
 * 
 */

// header {{{
#include <bits/stdc++.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <SDL.h>
#include <SDL_image.h>

using namespace std;

#define CPP_STR(x) CPP_STR_I(x)
#define CPP_CAT(x,y) CPP_CAT_I(x,y)
#define CPP_STR_I(args...) #args
#define CPP_CAT_I(x,y) x ## y

#define SFINAE(pred...) std::enable_if_t<(pred), std::nullptr_t> = nullptr

#define ASSERT(expr...) assert((expr))

using i8  = int8_t;
using u8  = uint8_t;
using i16 = int16_t;
using u16 = uint16_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

using f32  = float;
using f64  = double;
using f80  = __float80;
// }}}

// util {{{
#define FOR(i, start, end) for(int i = (start), CPP_CAT(i,xxxx_end)=(end); i < CPP_CAT(i,xxxx_end); ++i)
#define REP(i, n) FOR(i, 0, n)

#define ALL(f,c,args...) (([&](decltype((c)) cccc) { return (f)(std::begin(cccc), std::end(cccc), ## args); })(c))

template<typename C>
constexpr int SIZE(const C& c) {
    using std::size;
    return static_cast<int>(size(c));
}

template<typename T, size_t N>
constexpr int SIZE(const T (&)[N]) noexcept {
    return static_cast<int>(N);
}

template<typename... TS, typename F, size_t... IS>
void tuple_enumerate(tuple<TS...>& t, F&& f, index_sequence<IS...>) {
    (f(IS,get<IS>(t)), ...);
}

template<typename... TS, typename F>
void tuple_enumerate(tuple<TS...>& t, F&& f) {
    tuple_enumerate(t, f, index_sequence_for<TS...>{});
}

template<typename... TS, typename F, size_t... IS>
void tuple_enumerate(const tuple<TS...>& t, F&& f, index_sequence<IS...>) {
    (f(IS,get<IS>(t)), ...);
}

template<typename... TS, typename F>
void tuple_enumerate(const tuple<TS...>& t, F&& f) {
    tuple_enumerate(t, f, index_sequence_for<TS...>{});
}

template<typename T, typename U, typename Comp=less<>>
bool chmax(T& xmax, const U& x, Comp comp={}) {
    if(comp(xmax, x)) {
        xmax = x;
        return true;
    }
    return false;
}

template<typename T, typename U, typename Comp=less<>>
bool chmin(T& xmin, const U& x, Comp comp={}) {
    if(comp(x, xmin)) {
        xmin = x;
        return true;
    }
    return false;
}

template<typename C>
void UNIQ(C& c) {
    c.erase(ALL(unique,c), end(c));
}
// }}}

// debug {{{
template<typename T, typename Enable=void>
struct Repr {
    static ostream& write(ostream& out, const T& x) { return out << x; }
};

template<typename T>
ostream& REPR_WRITE(ostream& out, const T& x) {
    return Repr<T>::write(out, x);
}

template<typename T, size_t N>
ostream& REPR_WRITE(ostream& out, const T (&x)[N]) {
    return Repr<decltype(x)>::write(out, x);
}

template<typename InputIt>
ostream& REPR_WRITE_RANGE(ostream& out, InputIt first, InputIt last) {
    out << "[";
    while(first != last) {
        REPR_WRITE(out, *first++);
        if(first != last)
            out << ",";
    }
    out << "]";
    return out;
}

template<typename T>
string REPR(const T& x) {
    ostringstream out;
    REPR_WRITE(out, x);
    return out.str();
}

template<typename T, size_t N>
string REPR(const T (&x)[N]) {
    ostringstream out;
    REPR_WRITE(out, x);
    return out.str();
}

template<typename Enum>
struct Repr<Enum, enable_if_t<is_enum_v<Enum>>> {
    static ostream& write(ostream& out, Enum x) {
        return REPR_WRITE(out, static_cast<underlying_type_t<Enum>>(x));
    }
};

template<typename... TS>
struct Repr<tuple<TS...>> {
    static ostream& write(ostream& out, const tuple<TS...>& t) {
        out << "(";
        tuple_enumerate(t, [&out](int i, const auto& e) {
            if(i != 0) out << ",";
            REPR_WRITE(out, e);
        });
        out << ")";
        return out;
    }
};

template<typename T1, typename T2>
struct Repr<pair<T1,T2>> {
    static ostream& write(ostream& out, const pair<T1,T2>& p) {
        return REPR_WRITE(out, tuple<T1,T2>(p.first,p.second));
    }
};

template<typename T, size_t N>
struct Repr<T(&)[N], enable_if_t<!is_same_v<remove_cv_t<T>,char>>> {
    static ostream& write(ostream& out, const T (&x)[N]) {
        return REPR_WRITE_RANGE(out, begin(x), end(x));
    }
};

template<typename T> struct is_container : false_type {};
template<typename T, size_t N> struct is_container<array<T,N>> : true_type {};
template<typename... Args> struct is_container<vector<Args...>> : true_type {};
template<typename... Args> struct is_container<deque<Args...>> : true_type {};
template<typename... Args> struct is_container<list<Args...>> : true_type {};
template<typename... Args> struct is_container<forward_list<Args...>> : true_type {};
template<typename... Args> struct is_container<set<Args...>> : true_type {};
template<typename... Args> struct is_container<multiset<Args...>> : true_type {};
template<typename... Args> struct is_container<unordered_set<Args...>> : true_type {};
template<typename... Args> struct is_container<unordered_multiset<Args...>> : true_type {};
template<typename... Args> struct is_container<map<Args...>> : true_type {};
template<typename... Args> struct is_container<multimap<Args...>> : true_type {};
template<typename... Args> struct is_container<unordered_map<Args...>> : true_type {};
template<typename... Args> struct is_container<unordered_multimap<Args...>> : true_type {};

template<typename C>
struct Repr<C, enable_if_t<is_container<C>::value>> {
    static ostream& write(ostream& out, const C& c) {
        return REPR_WRITE_RANGE(out, begin(c), end(c));
    }
};

template<typename... TS>
void DBG_IMPL(int line, const char* expr, const tuple<TS...>& value) {
    cerr << "[L " << line << "]: ";
    if constexpr (sizeof...(TS) == 1) {
        cerr << expr << " = ";
        REPR_WRITE(cerr, get<0>(value));
    }
    else {
        cerr << "(" << expr << ") = ";
        REPR_WRITE(cerr, value);
    }
    cerr << "\n";
}

template<typename T, size_t N>
void DBG_CARRAY_IMPL(int line, const char* expr, const T (&ary)[N]) {
    cerr << "[L " << line << "]: ";
    cerr << expr << " = ";
    REPR_WRITE(cerr, ary);
    cerr << "\n";
}

template<typename InputIt>
void DBG_RANGE_IMPL(int line, const char* expr1, const char* expr2, InputIt first, InputIt last) {
    cerr << "[L " << line << "]: ";
    cerr << expr1 << "," << expr2 << " = ";
    REPR_WRITE_RANGE(cerr, first, last);
    cerr << "\n";
}

#define DBG_ENABLE
#ifdef DBG_ENABLE
    #define DBG(args...) DBG_IMPL(__LINE__, CPP_STR_I(args), std::make_tuple(args))
    #define DBG_CARRAY(expr) DBG_CARRAY_IMPL(__LINE__, CPP_STR(expr), (expr))
    #define DBG_RANGE(first,last) DBG_RANGE_IMPL(__LINE__, CPP_STR(first), CPP_STR(last), (first), (last))
#else
    #define DBG(args...)
    #define DBG_CARRAY(expr)
    #define DBG_RANGE(first,last)
#endif
// }}}

// random {{{

// PCGEngine {{{
// PCG random number generator
//
// http://www.pcg-random.org/
// https://github.com/imneme/pcg-c-basic
class PcgEngine {
public:
    using result_type = std::uint32_t;
    static constexpr result_type min() { return std::numeric_limits<result_type>::min(); }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    PcgEngine(std::uint64_t seed, std::uint64_t seq) {
        init(seed, seq);
    }

    template<typename SeedSeq>
    explicit PcgEngine(SeedSeq& ss) {
        std::uint32_t xs[4];
        ss.generate(std::begin(xs), std::end(xs));
        std::uint64_t seed = (std::uint64_t(xs[0])<<32) | xs[1];
        std::uint64_t seq  = (std::uint64_t(xs[2])<<32) | xs[3];
        init(seed, seq);
    }

    std::uint32_t operator()() {
        std::uint32_t xorshifted = std::uint32_t(((state_>>18) ^ state_) >> 27);
        std::uint32_t rot = std::uint32_t(state_ >> 59);
        std::uint32_t res = (xorshifted>>rot) | (xorshifted<<((-rot)&31));
        state_ = 6364136223846793005ULL*state_ + inc_;
        return res;
    }

private:
    void init(std::uint64_t seed, std::uint64_t seq) {
        state_ = 0;
        inc_ = (seq<<1) | 1;
        (*this)();
        state_ += seed;
        (*this)();
    }

    std::uint64_t state_;
    std::uint64_t inc_;
};
// }}}

// Seeder {{{
class Seeder {
private:
    static constexpr std::uint32_t INIT_A = 0x43b0d7e5;
    static constexpr std::uint32_t MULT_A = 0x931e8875;

    static constexpr std::uint32_t INIT_B = 0x8b51f9dd;
    static constexpr std::uint32_t MULT_B = 0x58f38ded;

    static constexpr std::uint32_t MIX_MULT_L = 0xca01f9dd;
    static constexpr std::uint32_t MIX_MULT_R = 0x4973f715;
    static constexpr std::uint32_t XSHIFT = 16;

    static constexpr std::uint32_t fast_exp(std::uint32_t x, std::uint32_t e) {
        std::uint32_t res = 1;
        std::uint32_t mul = x;
        while(e > 0) {
            if(e & 1)
                res *= mul;
            mul *= mul;
            e >>= 1;
        }
        return res;
    }

    template<typename InputIt>
    void mix_entropy(InputIt first, InputIt last) {
        std::uint32_t hash_const = INIT_A;
        auto h = [&hash_const](std::uint32_t x) {
            x ^= hash_const;
            hash_const *= MULT_A;
            x *= hash_const;
            x ^= x >> XSHIFT;
            return x;
        };
        auto mix = [](std::uint32_t x, std::uint32_t y) {
            std::uint32_t res = MIX_MULT_L*x - MIX_MULT_R*y;
            res ^= res >> XSHIFT;
            return res;
        };

        InputIt it = first;
        for(auto& elem : mixer_)
            elem = h(it == last ? 0 : *it++);
        for(auto& src : mixer_)
            for(auto& dest : mixer_)
                if(&src != &dest)
                    dest = mix(dest, h(src));
        for(; it != last; ++it)
            for(auto& dest : mixer_)
                dest = mix(dest, h(*it));
    }

    std::uint32_t mixer_[4];

public:
    using result_type = std::uint32_t;

    Seeder(const Seeder&)         = delete;
    void operator=(const Seeder&) = delete;

    template<typename InputIt>
    Seeder(InputIt first, InputIt last) {
        seed(first, last);
    }

    template<typename T>
    Seeder(std::initializer_list<T> ilist) : Seeder(std::begin(ilist),std::end(ilist)) {}

    template<typename InputIt>
    void seed(InputIt first, InputIt last) {
        mix_entropy(first, last);
    }

    template<typename RandomIt>
    void generate(RandomIt dst_first, RandomIt dst_last) const {
        auto src_first = std::begin(mixer_);
        auto src_last  = std::end(mixer_);
        auto src       = src_first;
        std::uint32_t hash_const = INIT_B;
        for(auto dst = dst_first; dst != dst_last; ++dst) {
            std::uint32_t x = *src++;
            if(src == src_last)
                src = src_first;
            x ^= hash_const;
            hash_const *= MULT_B;
            x *= hash_const;
            x ^= x >> XSHIFT;
            *dst = x;
        }
    }

    std::size_t size() const { return 4; }

    template<typename OutputIt>
    void param(OutputIt first) const {
        constexpr std::uint32_t INV_A     = fast_exp(MULT_A,     std::uint32_t(-1));
        constexpr std::uint32_t MIX_INV_L = fast_exp(MIX_MULT_L, std::uint32_t(-1));

        std::uint32_t res[4];
        std::copy(std::begin(mixer_), std::end(mixer_), std::begin(res));

        std::uint32_t hash_const = INIT_A * fast_exp(MULT_A, 16);
        for(auto src = std::rbegin(res); src != std::rend(res); ++src) {
            for(auto dst = std::rbegin(res); dst != std::rend(res); ++dst) {
                if(src == dst) continue;
                std::uint32_t revhashed = *src;
                std::uint32_t mult_const = hash_const;
                hash_const *= INV_A;
                revhashed ^= hash_const;
                revhashed *= mult_const;
                revhashed ^= revhashed >> XSHIFT;
                std::uint32_t unmixed = *dst;
                unmixed ^= unmixed >> XSHIFT;
                unmixed += MIX_MULT_R*revhashed;
                unmixed *= MIX_INV_L;
                *dst = unmixed;
            }
        }
        for(auto it = std::rbegin(res); it != std::rend(res); ++it) {
            std::uint32_t unhashed = *it;
            unhashed ^= unhashed >> XSHIFT;
            unhashed *= fast_exp(hash_const, std::uint32_t(-1));
            hash_const *= INV_A;
            unhashed ^= hash_const;
            *it = unhashed;
        }

        std::copy(std::begin(res), std::end(res), first);
    }
};
// }}}

// AutoSeeder {{{
#ifndef ENTROPY_CPU
    #if defined(__has_builtin)
        #if __has_builtin(__builtin_readcyclecounter)
            #define ENTROPY_CPU (__builtin_readcyclecounter())
        #endif
    #endif
#endif
#ifndef ENTROPY_CPU
    #if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
        #define ENTROPY_CPU (__builtin_ia32_rdtsc())
    #else
        #define ENTROPY_CPU (0)
    #endif
#endif

#ifndef ENTROPY_PID
    #if defined(__unix__) || defined(__unix) || defined(__APPLE__) || defined(__MACH__)
        #include <unistd.h>
        #define ENTROPY_PID (getpid())
    #else
        #define ENTROPY_PID (0)
    #endif
#endif

template<typename SeedSeq>
class AutoSeederT : public SeedSeq {
private:
    using EntropyArray = std::array<uint32_t,13>;

    template<typename T>
    static std::uint32_t crushto32(T x) {
        if(sizeof(T) <= 4) return std::uint32_t(x);

        std::uint64_t res = std::uint64_t(x);
        res *= 0xbc2ad017d719504d;
        return std::uint32_t(res ^ (res>>32));
    }

    template<typename T>
    static std::uint32_t hash_func(T&& x) {
        return crushto32(std::hash<std::remove_reference_t<std::remove_cv_t<T>>>()(std::forward<T>(x)));
    }

    static constexpr std::uint32_t fnv(std::uint32_t h, const char* pos) {
        if(*pos == '\0') return h;
        return fnv((16777619U*h) ^ *pos, pos+1);
    }

    EntropyArray entropy() const {
        constexpr std::uint32_t compile_stamp = fnv(2166136261U, __DATE__ __TIME__ __FILE__);

        static std::uint32_t random_int = std::random_device()();
        random_int += 0xedf19156;

        auto timestamp = crushto32(std::chrono::high_resolution_clock::now().time_since_epoch().count());

        void* malloc_ptr = std::malloc(sizeof(int));
        free(malloc_ptr);
        auto addr_heap  = hash_func(malloc_ptr);
        auto addr_stack = hash_func(&malloc_ptr);

        auto addr_this        = hash_func(this);
        auto addr_static_func = hash_func(static_cast<std::uint32_t (*)(std::uint64_t)>(&AutoSeederT::crushto32));

        auto addr_exit_func = hash_func(&_Exit);
        auto addr_time_func = hash_func(&std::chrono::high_resolution_clock::now);

        auto thread_id = hash_func(std::this_thread::get_id());

#if __cpp_rtti || __GXX_RTTI
        auto type_id = crushto32(typeid(*this).hash_code());
#else
        uint32_t type_id = 0;
#endif

        auto cpu = crushto32(ENTROPY_CPU);
        auto pid = crushto32(ENTROPY_PID);

        return {{
            compile_stamp,
            random_int,
            timestamp,
            addr_heap,
            addr_stack,
            addr_this,
            addr_static_func,
            addr_exit_func,
            addr_time_func,
            thread_id,
            type_id,
            cpu,
            pid,
        }};
    }

    AutoSeederT(EntropyArray ea) : SeedSeq(std::begin(ea),std::end(ea)) {}

public:
    AutoSeederT() : AutoSeederT(entropy()) {}

    const SeedSeq& base() const { return *this; }
          SeedSeq& base()       { return *this; }
};

using AutoSeeder = AutoSeederT<Seeder>;

#undef ENTROPY_CPU
#undef ENTROPY_PID
// }}}

// UniformDistributionType {{{
template<typename T>
struct IsUniformInt : std::integral_constant<bool,
    std::is_same<T,short             >::value ||
    std::is_same<T,int               >::value ||
    std::is_same<T,long              >::value ||
    std::is_same<T,long long         >::value ||
    std::is_same<T,unsigned short    >::value ||
    std::is_same<T,unsigned int      >::value ||
    std::is_same<T,unsigned long     >::value ||
    std::is_same<T,unsigned long long>::value
> {};

template<typename T>
struct IsUniformReal : std::integral_constant<bool,
    std::is_same<T,float      >::value ||
    std::is_same<T,double     >::value ||
    std::is_same<T,long double>::value
> {};

template<typename T>
struct IsUniformByte : std::integral_constant<bool,
    std::is_same<T,char         >::value ||
    std::is_same<T,signed char  >::value ||
    std::is_same<T,unsigned char>::value
> {};

template<typename T, typename Enable=void>
struct UniformDistribution {};

template<typename T>
struct UniformDistribution<T, std::enable_if_t<IsUniformInt<T>::value>> {
    using type = std::uniform_int_distribution<T>;
};

template<typename T>
struct UniformDistribution<T, std::enable_if_t<IsUniformReal<T>::value>> {
    using type = std::uniform_real_distribution<T>;
};

template<typename T>
using UniformDistributionType = typename UniformDistribution<T>::type;

template<typename T, typename Enable=void>
struct UniformDistributionImpl {};

template<typename T>
class UniformDistributionImpl<T, std::enable_if_t<IsUniformByte<T>::value>> {
private:
    using Short = std::conditional_t<std::is_signed<T>::value, short, unsigned short>;
    using Dist  = UniformDistributionType<Short>;

    Dist dist_;

public:
    using result_type = T;
    using param_type  = typename Dist::param_type;

    explicit UniformDistributionImpl(result_type a=0,
                                     result_type b=std::numeric_limits<result_type>::max())
        : dist_(a,b) {}
    explicit UniformDistributionImpl(const param_type& p)
        : dist_(p) {}
    void reset() {}

    template<typename URNG>
    result_type operator()(URNG& g) {
        return static_cast<result_type>(dist_(g));
    }
    template<typename URNG>
    result_type operator()(URNG& g, const param_type& p) {
        return static_cast<result_type>(dist_(g,p));
    }

    result_type a() const { return static_cast<result_type>(dist_.a()); }
    result_type b() const { return static_cast<result_type>(dist_.b()); }

    param_type param() const { return dist_.param(); }
    void param(const param_type& p) { dist_.param(p); }

    result_type min() const { return a(); }
    result_type max() const { return b(); }

    friend bool operator==(const UniformDistributionImpl& lhs,
                           const UniformDistributionImpl& rhs) {
        return lhs.dist_ == rhs.dist_;
    }
    friend bool operator!=(const UniformDistributionImpl& lhs,
                           const UniformDistributionImpl& rhs) {
        return !(lhs == rhs);
    }
};

template<>
class UniformDistributionImpl<bool> {
private:
    using Dist  = UniformDistributionType<int>;

    Dist dist_;

public:
    using result_type = bool;
    using param_type  = typename Dist::param_type;

    explicit UniformDistributionImpl(result_type a=false, result_type b=true)
        : dist_(a,b) {}
    explicit UniformDistributionImpl(const param_type& p)
        : dist_(p) {}
    void reset() {}

    template<typename URNG>
    result_type operator()(URNG& g) {
        return static_cast<result_type>(dist_(g));
    }
    template<typename URNG>
    result_type operator()(URNG& g, const param_type& p) {
        return static_cast<result_type>(dist_(g,p));
    }

    result_type a() const { return static_cast<result_type>(dist_.a()); }
    result_type b() const { return static_cast<result_type>(dist_.b()); }

    param_type param() const { return dist_.param(); }
    void param(const param_type& p) { dist_.param(p); }

    result_type min() const { return a(); }
    result_type max() const { return b(); }

    friend bool operator==(const UniformDistributionImpl& lhs,
                           const UniformDistributionImpl& rhs) {
        return lhs.dist_ == rhs.dist_;
    }
    friend bool operator!=(const UniformDistributionImpl& lhs,
                           const UniformDistributionImpl& rhs) {
        return !(lhs == rhs);
    }
};

template<typename T>
struct UniformDistribution<T, std::enable_if_t<IsUniformByte<T>::value>> {
    using type = UniformDistributionImpl<T>;
};
template<>
struct UniformDistribution<bool> {
    using type = UniformDistributionImpl<bool>;
};
// }}}

// Rng {{{
template<typename Engine>
class RngT {
private:
    Engine engine_;

public:
    RngT() : engine_(AutoSeeder().base()) {}
    explicit RngT(std::uint32_t seed) : engine_(Seeder({seed})) {}

    void seed(std::uint32_t seed) { engine_.seed(Seeder({seed})); }

    template<typename R,
             template<typename> class DistT=std::normal_distribution,
             typename... Params>
    R variate(Params&&... params) {
        DistT<R> dist(std::forward<Params>(params)...);
        return dist(engine_);
    }

    // [a,b]
    template<typename T1, typename T2,
             std::enable_if_t<
                 IsUniformInt<T1>::value &&
                 IsUniformInt<T2>::value &&
                 std::is_signed<T1>::value != std::is_unsigned<T2>::value,
                 std::nullptr_t
             > = nullptr>
    auto uniform(T1 a, T2 b) {
        using R = std::common_type_t<T1,T2>;
        return variate<R,UniformDistributionType>(a, b);
    }

    // [a,b]
    template<typename T,
             std::enable_if_t<IsUniformInt<T>::value, std::nullptr_t> = nullptr>
    T uniform(T a=std::numeric_limits<T>::min(),
              T b=std::numeric_limits<T>::max()) {
        return variate<T,UniformDistributionType>(a, b);
    }

    // [a,b)
    template<typename T1, typename T2,
             std::enable_if_t<
                 IsUniformReal<T1>::value &&
                 IsUniformReal<T2>::value,
                 std::nullptr_t
             > = nullptr>
    auto uniform(T1 a, T2 b) {
        using R = std::common_type_t<T1,T2>;
        return variate<R,UniformDistributionType>(a, b);
    }

    // [a,b)
    template<typename T,
             std::enable_if_t<IsUniformReal<T>::value, std::nullptr_t> = nullptr>
    T uniform(T a=-std::numeric_limits<T>::max(),
              T b= std::numeric_limits<T>::max()) {
        return variate<T,UniformDistributionType>(a, b);
    }

    // [a,b]
    template<typename T,
             std::enable_if_t<IsUniformByte<T>::value, std::nullptr_t> = nullptr>
    T uniform(T a=std::numeric_limits<T>::min(),
              T b=std::numeric_limits<T>::max()) {
        return variate<T,UniformDistributionType>(a, b);
    }

    // [a,b]
    template<typename T,
             std::enable_if_t<std::is_same<T,bool>::value, std::nullptr_t> = nullptr>
    T uniform(T a=false, T b=true) {
        return variate<T,UniformDistributionType>(a, b);
    }

    template<template<typename> class DistT=UniformDistributionType,
             typename OutputIt,
             typename... Params>
    void generate(OutputIt first, OutputIt last, Params&&... params) {
        using R = std::remove_reference_t<decltype(*first)>;
        DistT<R> dist(std::forward<Params>(params)...);
        std::generate(first, last, [this,&dist]() { return dist(engine_); });
    }

    template<typename RandomIt>
    void shuffle(RandomIt first, RandomIt last) {
        std::shuffle(first, last, engine_);
    }

    template<typename ForwardIt>
    ForwardIt choose(ForwardIt first, ForwardIt last) {
        auto d = std::distance(first, last);
        if(d < 2) return first;

        using distance_type = decltype(d);
        distance_type choice = uniform(distance_type(0), --d);
        return std::next(first, choice);
    }

    template<typename T>
    auto pick(std::initializer_list<T> ilist) -> decltype(*std::begin(ilist)) {
        return *choose(std::begin(ilist), std::end(ilist));
    }

    template<typename BidiIt>
    BidiIt sample(std::size_t size, BidiIt first, BidiIt last) {
        auto total = std::distance(first, last);
        using distance_type = decltype(total);
        return std::stable_partition(first, last, [this,&size,&total](const auto&) {
            --total;
            if(uniform(distance_type(0),total) < size) {
                --size;
                return true;
            }
            else {
                return false;
            }
        });
    }
};

using Rng = RngT<PcgEngine>;
// }}}

// }}}

//--------------------------------------------------------------------

[[noreturn]] void error(const string& msg) {
    cerr << msg << "\n";
    exit(1);
}

constexpr int N_KIND = 34;

constexpr int FIELD_HEI =  8;
constexpr int FIELD_WID = 17;
constexpr int FIELD_N   = FIELD_HEI * FIELD_WID;

constexpr int WIN_WID = 640;
constexpr int WIN_HEI = 480;

constexpr int ORIGIN_X = 32;
constexpr int ORIGIN_Y = 32;

constexpr int SPRITE_WID = 33;
constexpr int SPRITE_HEI = 46;
constexpr int SPRITE_MARGIN = 1;

enum class Path {
    H=0, V, HV, VH, HVH, VHV, NONE
};

template<>
struct Repr<Path> {
    static ostream& write(ostream& out, Path p) {
        switch(p) {
        case Path::H:    out << "H";    break;
        case Path::V:    out << "V";    break;
        case Path::HV:   out << "HV";   break;
        case Path::VH:   out << "VH";   break;
        case Path::HVH:  out << "HVH";  break;
        case Path::VHV:  out << "VHV";  break;
        case Path::NONE: out << "NONE"; break;
        default: ASSERT(false);
        }
        return out;
    }
};

Path path_combine(Path a, Path b) {
    static constexpr Path RES[7][7] {
        { Path::H,    Path::HV,   Path::HV,   Path::HVH,  Path::HVH,  Path::NONE, Path::NONE },
        { Path::VH,   Path::V,    Path::VHV,  Path::VH,   Path::NONE, Path::VHV,  Path::NONE },
        { Path::HVH,  Path::HV,   Path::NONE, Path::HVH,  Path::NONE, Path::NONE, Path::NONE },
        { Path::VH,   Path::VHV,  Path::VHV,  Path::NONE, Path::NONE, Path::NONE, Path::NONE },
        { Path::HVH,  Path::NONE, Path::NONE, Path::NONE, Path::NONE, Path::NONE, Path::NONE },
        { Path::NONE, Path::VHV,  Path::NONE, Path::NONE, Path::NONE, Path::NONE, Path::NONE },
        { Path::NONE, Path::NONE, Path::NONE, Path::NONE, Path::NONE, Path::NONE, Path::NONE },
    };

    return RES[int(a)][int(b)];
}

vector<Path> paths_combine(const vector<Path>& v1, const vector<Path>& v2) {
    vector<Path> res;
    for(Path p1 : v1) for(Path p2 : v2) {
        Path q = path_combine(p1, p2);
        if(q == Path::NONE) continue;
        res.emplace_back(q);
    }
    ALL(sort, res);
    UNIQ(res);
    return res;
}

array<int,FIELD_N> tiles;
array<array<vector<Path>,FIELD_N>,FIELD_N> graph;

SDL_Window* win;
SDL_Renderer* ren;
array<SDL_Texture*,N_KIND> texs;

bool running;
array<int,2> picks;
bool hint;

int rc2idx(int r, int c) {
    return FIELD_WID*r + c;
}

pair<int,int> idx2rc(int i) {
    return { i/FIELD_WID, i%FIELD_WID };
}

void field_init() {
    static Rng rng;

    REP(i, FIELD_N) {
        int kind = i / 4;
        tiles[i] = kind;
    }
    ALL(rng.shuffle, tiles);

    // 横隣接
    REP(r, FIELD_HEI) REP(c, FIELD_WID-1) {
        int s = rc2idx(r,c);
        int t = rc2idx(r,c+1);
        graph[s][t].emplace_back(Path::H);
        graph[t][s].emplace_back(Path::H);
    }

    // 縦隣接
    REP(r, FIELD_HEI-1) REP(c, FIELD_WID) {
        int s = rc2idx(r,c);
        int t = rc2idx(r+1,c);
        graph[s][t].emplace_back(Path::V);
        graph[t][s].emplace_back(Path::V);
    }

    // 上下外周
    REP(c1, FIELD_WID-1) FOR(c2, c1+1, FIELD_WID) {
        {
            int s = rc2idx(0,c1);
            int t = rc2idx(0,c2);
            graph[s][t].emplace_back(Path::VHV);
            graph[t][s].emplace_back(Path::VHV);
        }
        {
            int s = rc2idx(FIELD_HEI-1,c1);
            int t = rc2idx(FIELD_HEI-1,c2);
            graph[s][t].emplace_back(Path::VHV);
            graph[t][s].emplace_back(Path::VHV);
        }
    }

    // 左右外周
    REP(r1, FIELD_HEI-1) FOR(r2, r1+1, FIELD_HEI) {
        {
            int s = rc2idx(r1,0);
            int t = rc2idx(r2,0);
            graph[s][t].emplace_back(Path::HVH);
            graph[t][s].emplace_back(Path::HVH);
        }
        {
            int s = rc2idx(r1,FIELD_WID-1);
            int t = rc2idx(r2,FIELD_WID-1);
            graph[s][t].emplace_back(Path::HVH);
            graph[t][s].emplace_back(Path::HVH);
        }
    }
}

void field_rm(int s) {
    REP(t1, FIELD_N) REP(t2, FIELD_N) {
        if(t1 == t2) continue;
        if(t1 == s || t2 == s) continue;

        auto& ps = graph[t1][t2];
        auto ps_append = paths_combine(graph[t1][s], graph[s][t2]);
        ps.insert(end(ps), begin(ps_append), end(ps_append));
        ALL(sort, ps);
        UNIQ(ps);
    }

    tiles[s] = -1;
}

void field_mv(int s, int t) {
    ASSERT(tiles[s] == tiles[t]);
    ASSERT(!graph[s][t].empty());

    field_rm(s);
    field_rm(t);
}

void picks_init() {
    ALL(fill, picks, -1);
}

bool field_mv_ok(int s, int t) {
    ASSERT(s != t);
    if(tiles[s] == -1 || tiles[t] == -1) return false;
    if(tiles[s] != tiles[t]) return false;
    return !graph[s][t].empty();
}

void game_init() {
    field_init();
    picks_init();
    hint = false;
}

pair<int,int> rc2xy(int r, int c) {
    return {
        ORIGIN_X + (SPRITE_WID+SPRITE_MARGIN)*c,
        ORIGIN_Y + (SPRITE_HEI+SPRITE_MARGIN)*r,
    };
}

pair<int,int> xy2rc(int x, int y) {
    x -= ORIGIN_X;
    y -= ORIGIN_Y;
    if(x < 0 || y < 0) return { -1, -1 };

    int r = y / (SPRITE_HEI+SPRITE_MARGIN);
    int c = x / (SPRITE_WID+SPRITE_MARGIN);
    if(r >= FIELD_HEI || c >= FIELD_WID) return { -1, -1 };

    return { r, c };
}

void on_quit(const SDL_QuitEvent&) {
    running = false;
}

void on_keydown(const SDL_KeyboardEvent& key) {
    switch(key.keysym.sym) {
    case SDLK_h:
        hint = !hint;
        break;
    case SDLK_q:
        running = false;
        break;
    case SDLK_r:
        game_init();
        break;
    default: break;
    }
}

void on_mousedown(const SDL_MouseButtonEvent& button) {
    switch(button.button) {
    case SDL_BUTTON_LEFT: {
        int i = int(distance(begin(picks), ALL(find,picks,-1)));
        if(i == 2) return;

        auto [r,c] = xy2rc(button.x, button.y);
        if(r == -1) return;
        int s = rc2idx(r, c);

        if(tiles[s] == -1) return;
        if(i == 1) {
            if(s == picks[0]) return;
            if(tiles[s] != tiles[picks[0]]) return;
        }

        picks[i] = s;
        break;
    }
    case SDL_BUTTON_RIGHT:
        picks_init();
        break;
    default: break;
    }
}

void process_events() {
    SDL_Event ev;
    while(SDL_PollEvent(&ev) != 0) {
        switch(ev.type) {
        case SDL_QUIT: on_quit(ev.quit); break;
        case SDL_KEYDOWN: on_keydown(ev.key); break;
        case SDL_MOUSEBUTTONDOWN: on_mousedown(ev.button); break;
        default: break;
        }
    }
}

void update() {
    if(ALL(find, picks, -1) != end(picks)) return;

    int s = picks[0];
    int t = picks[1];
    if(field_mv_ok(s, t)) {
        field_mv(s, t);
    }
    picks_init();
}

void draw() {
    SDL_SetRenderDrawColor(ren, 0x00, 0x00, 0x00, 0xFF);
    SDL_RenderClear(ren);

    REP(r, FIELD_HEI) REP(c, FIELD_WID) {
        int kind = tiles[rc2idx(r,c)];
        if(kind == -1) continue;

        auto [x,y] = rc2xy(r, c);
        SDL_Texture* tex = texs[kind];
        SDL_Rect dst { x, y, SPRITE_WID, SPRITE_HEI };

        auto it = ALL(find, picks, rc2idx(r,c));
        u8 alpha = it == end(picks) ? 0xFF : 0x80;
        SDL_SetTextureAlphaMod(tex, alpha);
        SDL_RenderCopy(ren, tex, nullptr, &dst);
    }

    if(hint) {
        SDL_SetRenderDrawColor(ren, 0x00, 0x00, 0xFF, 0xFF);

        REP(s, FIELD_N) FOR(t, s+1, FIELD_N) {
            if(field_mv_ok(s, t)) {
                auto [r1,c1] = idx2rc(s);
                auto [r2,c2] = idx2rc(t);
                auto [x1,y1] = rc2xy(r1,c1);
                auto [x2,y2] = rc2xy(r2,c2);
                x1 += SPRITE_WID / 2;
                y1 += SPRITE_HEI / 2;
                x2 += SPRITE_WID / 2;
                y2 += SPRITE_HEI / 2;
                SDL_RenderDrawLine(ren, x1, y1, x2, y2);
            }
        }
    }
}

void mainloop() {
    game_init();

    for(running = true; running; ) {
        process_events();

        update();

        draw();
        SDL_RenderPresent(ren);

        SDL_Delay(16);
    }
}

int main() {
    if(SDL_Init(SDL_INIT_VIDEO) < 0)
        error("SDL_Init() failed");

    {
        int flags = IMG_INIT_PNG;
        int res = IMG_Init(flags);
        if((res&flags) != flags)
            error("IMG_Init() failed");
    }

    win = SDL_CreateWindow(
        /* title= */ "shisen",
        /* x=     */ SDL_WINDOWPOS_UNDEFINED,
        /* y=     */ SDL_WINDOWPOS_UNDEFINED,
        /* w=     */ WIN_WID,
        /* h=     */ WIN_HEI,
        /* flags= */ 0
    );
    if(!win) error("SDL_CreateWindow() failed");

    ren = SDL_CreateRenderer(
        /* window= */ win,
        /* index=  */ -1,
        /* flags=  */ 0
    );
    if(!ren) error("SDL_CreateRenderer() failed");

    {
        REP(i, N_KIND) {
            SDL_Surface* surf = IMG_Load(fmt::format("asset/{:02}.png",i).c_str());
            if(!surf) error("IMG_Load() failed");
            SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
            if(!tex) error("SDL_CreateTextureFromSurface() failed");
            if(SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND) < 0)
                error("SDL_SetTextureBlendMode() failed");
            texs[i] = tex;
            SDL_FreeSurface(surf);
        }
    }

    mainloop();

    ALL(for_each, texs, SDL_DestroyTexture);

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    IMG_Quit();
    SDL_Quit();

    return 0;
}
