//===- IRDLContext.cpp - IRDL context ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IRDLContext.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace irdl;

/// A wrapper around the 'buitin.complex' type.
class ComplexTypeWrapper : public CppTypeWrapper<ComplexType> {
  StringRef getName() override { return "builtin.complex"; }

  SmallVector<Attribute> getTypeParameters(ComplexType type) override {
    return {TypeAttr::get(type.getElementType())};
  }

  size_t getParameterAmount() override { return 1; }

  Type instantiateType(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<Attribute> parameters) override {
    if (parameters.size() != this->getParameterAmount()) {
      emitError().append("invalid number of type parameters ",
                         parameters.size(), " (expected ",
                         this->getParameterAmount(), ")");
      return Type();
    }

    return ComplexType::getChecked(emitError,
                                   cast<TypeAttr>(parameters[0]).getValue());
  }
};

IRDLContext::IRDLContext() { addTypeWrapper<ComplexTypeWrapper>(); }

void IRDLContext::addAttributeWrapper(
    std::unique_ptr<AttributeWrapper> wrapper) {
  auto emplaced =
      typeIDToAttributeWrapper.insert({wrapper->getTypeID(), wrapper.get()});
  assert(emplaced.second &&
         "an attribute wrapper with the same name already exists");
  attributes.try_emplace(wrapper->getName(), std::move(wrapper));
}

AttributeWrapper *IRDLContext::getAttributeWrapper(StringRef attrName) {
  auto it = attributes.find(attrName);
  if (it == attributes.end())
    return nullptr;
  return it->second.get();
}

void IRDLContext::addTypeWrapper(std::unique_ptr<TypeWrapper> wrapper) {
  auto emplaced =
      typeIDToTypeWrapper.insert({wrapper->getTypeID(), wrapper.get()});
  assert(emplaced.second && "a type wrapper with the same name already exists");
  types.try_emplace(wrapper->getName(), std::move(wrapper));
}

TypeWrapper *IRDLContext::getTypeWrapper(StringRef typeName) {
  auto it = types.find(typeName);
  if (it == types.end())
    return nullptr;
  return it->second.get();
}
