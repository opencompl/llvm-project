#ifndef IRDLTOCPP_TEMPLATE_UTILS_H
#define IRDLTOCPP_TEMPLATE_UTILS_H

#include "llvm/ADT/SmallVector.h"
#include <string>
#include <array>

namespace mlir::irdl::detail {
struct DialectStrings {
  std::string dialectName;
  std::string dialectCppName;
  std::string dialectCppShortName;
  std::string dialectBaseTypeName;

  std::string namespaceOpen;
  std::string namespaceClose;
  std::string namespacePath;
};

struct TypeStrings {
  std::string typeName;
  std::string typeCppName;
};

struct OpStrings {
  StringRef opName;
  std::string opCppName;
  llvm::SmallVector<std::string> opResultNames;
  llvm::SmallVector<std::string> opOperandNames;
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

template<typename T1, typename T2, size_t N>
constexpr auto map_first(const std::pair<T1, T2> (&arrs)[N]) {
    std::array<T1, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = arrs[i].first;
    return ret; 
}

template<typename T1, typename T2, size_t N>
constexpr auto map_second(const std::pair<T1, T2> (&arrs)[N]) {
    std::array<T1, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = arrs[i].second;
    return ret; 
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

struct property_getter
{
    constexpr property_getter(std::string(* fn)(const DialectStrings&)) {}
    constexpr property_getter(std::string(* fn)(const TypeStrings&)) {}
    constexpr property_getter(std::string(* fn)(const OpStrings&)) {}
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
    constexpr std::pair<std::string_view, property_getter> variables[] = {
        #include "TemplatingVars.h"
    };
    constexpr auto known_ids = map_first(variables);
    constexpr auto varCount = std::size(variables);

    std::array<token_string, varCount> index_lookup_table {};

    uint64_t found_ids{};
    {
        char prevToken = '\0';
        bool isScanning = false;
        size_t scanStart = 0;
        for (size_t r = 0, w = 0; r < N; ++r) {
            const auto currToken = workingString[r];
            if (prevToken == '_' && currToken == '_') {
                if (!isScanning) {
                    // transit to scanning
                    scanStart = r-1;
                    isScanning = true;
                } else {
                    // handle scan name
                    std::string_view name {workingString.data() + scanStart, r - scanStart + 1};

                    size_t i{};
                    for (i = 0; i < varCount; ++i) {
                        if (known_ids[i] == name) 
                            break;
                    }

                    if (i < varCount) {
                        auto flag = 1ULL << i;

                        if (!(found_ids & flag)) {
                            // token is new
                            int pos = count_set_bits(found_ids);
                            index_lookup_table[i] = int_to_replace_token(pos);
                            found_ids |= flag;
                        } 

                        // replace the scan with the replacement token
                        auto replacement_token = index_lookup_table[i];
                        
                        w = scanStart;
                        for (auto c : replacement_token)
                            if (c)
                                workingString[w++] = c;
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
    }

    return std::make_pair(workingString, found_ids);
}
} // namespace mlir::irdl

#endif // #ifndef IRDLTOCPP_TEMPLATE_UTILS_H