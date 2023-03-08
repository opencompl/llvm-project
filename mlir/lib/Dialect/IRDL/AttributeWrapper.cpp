//===- AttributeWrapper.cpp - IRDL type wrapper definition ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/AttributeWrapper.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"

namespace mlir {
namespace irdl {

bool TypeWrapper::isCorrectAttribute(Attribute attr) {
  return isa<TypeAttr>(attr) && isCorrectType(cast<TypeAttr>(attr).getValue());
}

SmallVector<Attribute> TypeWrapper::getAttributeParameters(Attribute attr) {
  return getTypeParameters(cast<TypeAttr>(attr).getValue());
}

Attribute
TypeWrapper::instantiateAttribute(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<Attribute> parameters) {
  Type type = instantiateType(emitError, parameters);
  if (!type)
    return {};
  return TypeAttr::get(type);
}

} // namespace irdl
} // namespace mlir
