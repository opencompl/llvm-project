//===- IRDLConstraint.cpp - IRDL constraints definition ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the different type constraints an operand or a result can
// have.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IRDLConstraint.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"

using namespace mlir;
using namespace irdl;

//===----------------------------------------------------------------------===//
// Explicit instanciations
//===----------------------------------------------------------------------===//

namespace mlir {
namespace irdl {

template class EqConstraint<Type>;
template class EqConstraint<Attribute>;

template class AnyOfConstraint<Type>;
template class AnyOfConstraint<Attribute>;

template class AndConstraint<Type>;
template class AndConstraint<Attribute>;

template class VarConstraint<Type>;
template class VarConstraint<Attribute>;

} // namespace irdl
} // namespace mlir

//===----------------------------------------------------------------------===//
// Variable stores
//===----------------------------------------------------------------------===//

template <>
IRDLConstraint<Type> const &
VarConstraints::getVariableConstraint(size_t id) const {
  assert(id < typeConstr.size() &&
         "type constraint variable index out of bounds");
  return *typeConstr[id].get();
}

template <>
IRDLConstraint<Attribute> const &
VarConstraints::getVariableConstraint(size_t id) const {
  assert(id < attrConstr.size() &&
         "attribute constraint variable index out of bounds");
  return *attrConstr[id].get();
}

template <>
Type VarStore::getVariableValue(size_t id) const {
  assert(id < typeValues.size() &&
         "type constraint variable index out of bounds");
  return typeValues[id];
}

template <>
void VarStore::setVariableValue(size_t id, Type val) {
  assert(id < typeValues.size() &&
         "type constraint variable index out of bounds");
  typeValues[id] = val;
}

template <>
Attribute VarStore::getVariableValue(size_t id) const {
  assert(id < attrValues.size() &&
         "attribute constraint variable index out of bounds");
  return attrValues[id];
}

template <>
void VarStore::setVariableValue(size_t id, Attribute val) {
  assert(id < attrValues.size() &&
         "attribute constraint variable index out of bounds");
  attrValues[id] = val;
}

//===----------------------------------------------------------------------===//
// Constraints
//===----------------------------------------------------------------------===//

template <class Item>
LogicalResult EqConstraint<Item>::verify(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Item item,
    VarConstraints const &cstrs, VarStore &store) const {
  if (item == expectedItem)
    return success();

  if (emitError)
    return (*emitError)().append("expected ", expectedItem, " but got ", item);
  return failure();
}

template <class Item>
LogicalResult AnyOfConstraint<Item>::verify(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Item item,
    VarConstraints const &cstrs, VarStore &store) const {
  for (auto &constr : constrs) {
    VarStore newVarStore =
        store; // TODO: @reviewer: is this efficient (vector cloning aside)?
    if (succeeded(constr->verify({}, item, cstrs, newVarStore))) {
      store = newVarStore;
      return success();
    }
  }

  if (emitError)
    return (*emitError)().append(item, " does not satisfy the constraint");
  return failure();
}

template <class Item>
LogicalResult AndConstraint<Item>::verify(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Item item,
    VarConstraints const &cstrs, VarStore &store) const {
  for (auto &constr : constrs) {
    if (failed(constr->verify({}, item, cstrs, store))) {
      if (emitError)
        return (*emitError)().append(item, " does not satisfy the constraint ");
      return failure();
    }
  }

  return success();
}

template <class Item>
LogicalResult VarConstraint<Item>::verify(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Item item,
    VarConstraints const &cstrs, VarStore &store) const {
  // We first check if the variable was already assigned.
  auto expectedItem = store.getVariableValue<Item>(varIndex);
  if (expectedItem) {
    // If it is assigned, we check that our item is equal. If it is, we already
    // know we satisfy the underlying constraint.
    if (item == expectedItem) {
      return success();
    } else {
      if (emitError)
        return (*emitError)().append("expected ", expectedItem, " but got ",
                                     item);
      return failure();
    }
  }

  // We check that the type satisfies the type variable.
  IRDLConstraint<Item> const &constraint =
      cstrs.getVariableConstraint<Item>(varIndex);
  if (failed(constraint.verify(emitError, item, cstrs, store)))
    return failure();

  // At this point the item has been picked to be the definitive value
  // of the variable.
  store.setVariableValue<Item>(varIndex, item);

  return success();
}

LogicalResult TypeBaseConstraint::verify(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    VarConstraints const &cstrs, VarStore &store) const {
  if (typeDef->isCorrectType(type))
    return success();

  if (emitError)
    return (*emitError)().append("expected base type ", typeDef->getName(),
                                 ", but got ", type, " type.");
  return failure();
}

LogicalResult DynTypeBaseConstraint::verify(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    VarConstraints const &cstrs, VarStore &store) const {
  auto dynType = type.dyn_cast<DynamicType>();
  if (!dynType || dynType.getTypeDef() != dynTypeDef) {
    if (emitError)
      return (*emitError)().append(
          "expected base type '", dynTypeDef->getDialect()->getNamespace(), ".",
          dynTypeDef->getName(), "' but got type ", type);
    return failure();
  }
  return success();
}

LogicalResult DynTypeParamsConstraint::verify(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    VarConstraints const &cstrs, VarStore &store) const {
  auto dynType = type.dyn_cast<DynamicType>();
  if (!dynType || dynType.getTypeDef() != dynTypeDef) {
    if (emitError)
      return (*emitError)().append(
          "expected base type '", dynTypeDef->getDialect()->getNamespace(), ".",
          dynTypeDef->getName(), "' but got type ", type);
    return failure();
  }

  // Since we do not have variadic parameters yet, we should have the
  // exact number of constraints.
  assert(dynType.getParams().size() == paramConstraints.size() &&
         "unexpected number of parameters in parameter type constraint");
  auto params = dynType.getParams();
  for (size_t i = 0; i < params.size(); i++) {
    auto paramType = params[i].cast<TypeAttr>().getValue();
    if (failed(paramConstraints[i]->verify(emitError, paramType, cstrs, store)))
      return failure();
  }

  return success();
}

LogicalResult TypeParamsConstraint::verify(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    VarConstraints const &cstrs, VarStore &store) const {
  if (!typeDef->isCorrectType(type)) {
    if (emitError)
      return (*emitError)().append("expected base type '", typeDef->getName(),
                                   "' but got type ", type);
    return failure();
  }

  auto params = typeDef->getParameters(type);
  // Since we do not have variadic parameters yet, we should have the
  // exact number of constraints.
  assert(params.size() == paramConstraints.size() &&
         "unexpected number of parameters in parameter type constraint");
  for (size_t i = 0; i < params.size(); i++) {
    auto paramType = params[i].cast<TypeAttr>().getValue();
    if (failed(paramConstraints[i]->verify(emitError, paramType, cstrs, store)))
      return failure();
  }

  return success();
}
