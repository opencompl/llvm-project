#ifndef IRDLTOCPP_TEMPLATE_UTILS_H
#define IRDLTOCPP_TEMPLATE_UTILS_H

#include "llvm/ADT/SmallVector.h"
#include <string>
#include <array>

namespace mlir::irdl::detail {
using dictionary = llvm::StringMap<std::string>;

inline std::string formatTemplate(std::string_view value, const dictionary& dict) {
    std::string workingString;
    workingString.reserve(value.length() * 2);

    char prevToken = '\0';
    bool isProcessingVarName = false;
    size_t tokenStart{};
    
    for (size_t r = 0; r < value.length(); ++r)
    {        
        const auto currToken = value[r];
        if (currToken == '_' && prevToken == '_') {
            if (!isProcessingVarName) {
                tokenStart = r-1;
                isProcessingVarName = true;
            }
            else {
                llvm::StringRef varName {value.begin() + tokenStart + 2, r - tokenStart - 3};
                if (const auto itr = dict.find(varName); itr != dict.end()) {
                    const auto& value = itr->second;
                    workingString.resize(workingString.length() - 1);
                    workingString.insert(workingString.end(), value.begin(), value.end());
                } else {
                    llvm::errs() << "undefined variable: " << varName << "\n";
                    assert(false && "undefined variable");
                }
                isProcessingVarName = false;
                continue;
            }
        }

        if (!isProcessingVarName) 
            workingString.push_back(currToken);
        
        prevToken = currToken;
    }

    return workingString;
}
} // namespace mlir::irdl

#endif // #ifndef IRDLTOCPP_TEMPLATE_UTILS_H