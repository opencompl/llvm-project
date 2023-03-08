//===- IRDLContext.h - IRDL context -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration context of IRDL dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IRDLCONTEXT_H_
#define MLIR_DIALECT_IRDL_IRDLCONTEXT_H_

#include "mlir/Dialect/IRDL/AttributeWrapper.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace irdl {

/// Context for the runtime registration of IRDL dialect definitions.
/// This class keeps track of all the attribute and types defined in C++
/// that can be used in IRDL.
class IRDLContext {
  llvm::StringMap<std::unique_ptr<AttributeWrapper>> attributes;
  llvm::StringMap<std::unique_ptr<TypeWrapper>> types;
  DenseMap<TypeID, AttributeWrapper *> typeIDToAttributeWrapper;
  DenseMap<TypeID, TypeWrapper *> typeIDToTypeWrapper;

public:
  IRDLContext();

  /// Add a concrete attribute wrapper to IRDL.
  /// The attribute definition wrapped can then be used in IRDL with its name.
  template <typename A>
  void addAttributeWrapper() {
    addAttributeWrapper(std::make_unique<A>());
  }

  /// Add an attribute wrapper to IRDL.
  /// The attribute definition wrapped can then be used in IRDL with its name.
  void addAttributeWrapper(std::unique_ptr<AttributeWrapper> wrapper);

  AttributeWrapper *getAttributeWrapper(StringRef typeName);
  AttributeWrapper *getAttributeWrapper(TypeID typeID);

  /// Add a concrete type wrapper to IRDL.
  /// The type definition wrapped can then be used in IRDL with its name.
  template <typename T>
  void addTypeWrapper() {
    addTypeWrapper(std::make_unique<T>());
  }

  /// Add a type wrapper to IRDL.
  /// The type definition wrapped can then be used in IRDL with its name.
  void addTypeWrapper(std::unique_ptr<TypeWrapper> wrapper);

  TypeWrapper *getTypeWrapper(StringRef typeName);
  TypeWrapper *getTypeWrapper(TypeID typeID);

  llvm::StringMap<std::unique_ptr<AttributeWrapper>> const &getAllAttributes() {
    return attributes;
  }

  llvm::StringMap<std::unique_ptr<TypeWrapper>> const &getAllTypes() {
    return types;
  }
};

} // namespace irdl
} // namespace mlir

#endif // MLIR_DIALECT_IRDL_IRDLCONTEXT_H_
