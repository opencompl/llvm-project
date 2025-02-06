#ifndef IRDLTOCPP_TEMPLATE_UTILS_H
#define IRDLTOCPP_TEMPLATE_UTILS_H

#include "llvm/ADT/SmallVector.h"
#include <string>
#include <array>

namespace mlir::irdl::detail {
static constexpr std::string_view templating_variables[] = {
    #include "TemplatingVars.h"
};

template<typename ...Args>
constexpr auto count_args(Args&& ... args) {
    return sizeof...(Args);
}

constexpr auto count_set_bits(uint64_t flags) {
    size_t count = 0;
    for (size_t i = 0; i < sizeof(size_t) * 8; ++i)
        if (flags & (1ULL << i))
            ++count;
    return count;
}


using token_string = std::array<char, 5>;
constexpr auto int_to_replace_token(int i) -> token_string {
    assert(i < 64 && "only support less than 64 variables");
    if (i < 10) {
        token_string ret  {'{', '0', '}', '\0'};
        ret[1] += i;
        return ret;
    }
    if (i < 100) {
        token_string ret  {'{', '0', '0', '}', '\0'};
        ret[1] += i / 10;
        ret[1] += i % 10;
        return ret;
    }
    return {};
}

constexpr size_t tv_index(std::string_view label) {
    constexpr auto varCount = std::size(templating_variables);
    for (size_t i = 0; i < varCount; ++i) {
        if (templating_variables[i] == label) 
            return i;
    }
    return -1;
}

class tv_dictionary {
public:
    template<size_t TvIndex>
    void set(llvm::StringRef value) {
        static_assert(TvIndex >= 0 && TvIndex < std::size(templating_variables) && "unrecognized variable");
        dictionary[TvIndex] = value;
    }

    template<size_t TvIndex>
    llvm::StringRef get() const {
        return dictionary[TvIndex];
    }
private:
    std::array<std::string, std::size(templating_variables)> dictionary;
};

template<size_t Flags>
class template_formatter {
public:
    static constexpr auto var_count = count_set_bits(Flags);

    constexpr template_formatter(std::string_view fmtStr)
        : formatString{fmtStr}
    {}

    auto apply(const tv_dictionary& dict) const {
        return apply_impl(dict, std::make_index_sequence<var_count>{});
    }
private:
    std::string_view formatString;

    template<size_t Ind>
    static constexpr size_t get_tv_index() {
        int counter{};
        for (size_t i = 0; i < sizeof(Flags) * 8ULL; ++i) {
            if (Flags & (1ULL << i))
            {
                if (counter++ == Ind)
                    return i;
            }
        }
        return -1;
    }

    template<size_t ...Indexes>
    auto apply_impl(const tv_dictionary& dict, std::index_sequence<Indexes...>) const {
        return llvm::formatv(formatString.data(), dict.get<template_formatter::get_tv_index<Indexes>()>() ...);
    }
};

template<auto N>
constexpr auto process(const char (&ref)[N]) {
    std::array<char, N*2> workingString{};

    // copy string and escape braces
    for (size_t r = 0, w = 0; r < N; ++r, ++w) {
        workingString[w] = ref[r];
        if (ref[r] == '{')
            workingString[++w] = '{';
    }
    
    // collect all IDs, count unique IDs up to 64
    constexpr auto varCount = std::size(templating_variables);

    std::array<token_string, varCount> index_lookup_table {};

    uint64_t found_ids{};
    {
        std::string_view searchStr {ref};
        for (size_t i = 0; i < varCount; ++i) {
            const auto& tv = templating_variables[i];
            if (searchStr.find(tv) != std::string_view::npos) {
                int pos = count_set_bits(found_ids);
                index_lookup_table[i] = int_to_replace_token(pos);
                found_ids |= (1ULL << i);
            }
        }
    }
    {
        char prevToken = '\0';
        bool isScanning = false;
        size_t scanStart = 0;
        size_t w = 0;
        for (size_t r = 0; r < N; ++r) {
            const auto currToken = workingString[r];
            if (prevToken == '_' && currToken == '_') {
                if (!isScanning) {
                    // transit to scanning
                    scanStart = r-1;
                    isScanning = true;
                } else {
                    // handle scan name
                    std::string_view name {workingString.data() + scanStart, r - scanStart + 1};

                    size_t i = tv_index(name);

                    if (i < varCount) {
                        auto flag = 1ULL << i;

                        // replace the scan with the replacement token
                        auto replacement_token = index_lookup_table[i];
                        
                        w = w - 1;
                        for (auto c : replacement_token) {
                            if (c) {
                                workingString[w++] = c;
                            } else 
                                break;
                        }
                        isScanning = false;
                        continue;
                    }
                    else {
                        assert(false && "unknown id located"); // TODO: error string here
                    }
                }   
            } 

            if (!isScanning)  
                workingString[w++] = workingString[r];

            prevToken = currToken;
        }
        workingString[w] = 0;
    }

    return std::make_pair(workingString, found_ids);
}
} // namespace mlir::irdl

#endif // #ifndef IRDLTOCPP_TEMPLATE_UTILS_H