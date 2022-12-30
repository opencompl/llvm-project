//===- IRDLConstraint.h - IRDL constraints definition -----------*- C++ -*-===//
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

#ifndef MLIR_DIALECT_IRDL_IR_IRDLCONSTRAINT_H_
#define MLIR_DIALECT_IRDL_IR_IRDLCONSTRAINT_H_

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/TypeWrapper.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Hashing.h"

namespace mlir {
namespace irdl {
// Forward declaration.
class OperationOp;
template <class Item>
class Constraint;

/// Stores the definition of constraint variables with their associated
/// constraints.
///
/// Each kind of item has its own inner variable store, meaning that variable
/// indices are counted separately for types and attributes, for example.
class VarConstraints {
public:
  VarConstraints(
      ArrayRef<std::unique_ptr<Constraint<Type>>> typeConstr,
      ArrayRef<std::unique_ptr<Constraint<Attribute>>> attrConstr)
      : typeConstr(typeConstr), attrConstr(attrConstr) {}

  /// Returns the value of a constraint variable. Returns an empty
  /// item if the value is not yet set.
  template <class Item>
  Constraint<Item> const &getVariableConstraint(size_t id) const;

private:
  ArrayRef<std::unique_ptr<Constraint<Type>>> typeConstr;
  ArrayRef<std::unique_ptr<Constraint<Attribute>>> attrConstr;
};

/// Stores the value of constraint variables during verification.
///
/// Each kind of item has its own inner variable store, meaning that variable
/// indices are counted separately for types and attributes, for example.
class VarStore {
public:
  VarStore(size_t typeVarAmount, size_t attrVarAmount)
      : typeValues(typeVarAmount), attrValues(attrVarAmount) {}

  template <class Item>
  Item getVariableValue(size_t id) const;

  template <class Item>
  void setVariableValue(size_t id, Item val);

private:
  SmallVector<Type> typeValues;
  SmallVector<Attribute> attrValues;
};

/// A generic type constraint.
template <class Item>
class Constraint {
public:
  /// Check that an item is satisfying the constraint.
  /// `cstrs` are the constraints associated to the variables. They
  /// are accessed by their index.
  /// `store` contains the values of the constraint variables that are already
  /// defined, or an empty item if the value is not set yet.
  virtual LogicalResult
  verify(Optional<function_ref<InFlightDiagnostic()>> emitError, Item item,
         VarConstraints const &cstrs, VarStore &store) const = 0;

  virtual ~Constraint(){};
};

using TypeConstraint = Constraint<Type>;
using AttrConstraint = Constraint<Attribute>;

//===----------------------------------------------------------------------===//
// Equality constraint
//===----------------------------------------------------------------------===//

template <class Item>
class EqConstraint : public Constraint<Item> {
public:
  EqConstraint(Item expectedItem) : expectedItem(expectedItem) {}

  virtual LogicalResult
  verify(Optional<function_ref<InFlightDiagnostic()>> emitError, Item item,
         VarConstraints const &cstrs, VarStore &store) const override;

private:
  Item expectedItem;
};

//===----------------------------------------------------------------------===//
// AnyOf type constraint
//===----------------------------------------------------------------------===//

/// AnyOf constraint.
/// An item satisfies this constraint if it is included in a set of items.
template <class Item>
class AnyOfConstraint : public Constraint<Item> {
public:
  AnyOfConstraint(SmallVector<std::unique_ptr<Constraint<Item>>> constrs)
      : constrs(std::move(constrs)) {}

  virtual LogicalResult
  verify(Optional<function_ref<InFlightDiagnostic()>> emitError, Item item,
         VarConstraints const &cstrs, VarStore &store) const override;

private:
  llvm::SmallVector<std::unique_ptr<Constraint<Item>>> constrs;
};

//===----------------------------------------------------------------------===//
// And constraint
//===----------------------------------------------------------------------===//

/// And constraint.
/// An item satisfies this constraint if it satisfies a set of constraints.
template <class Item>
class AndConstraint : public Constraint<Item> {
public:
  AndConstraint(SmallVector<std::unique_ptr<Constraint<Item>>> constrs)
      : constrs(std::move(constrs)) {}

  virtual LogicalResult
  verify(Optional<function_ref<InFlightDiagnostic()>> emitError, Item item,
         VarConstraints const &cstrs, VarStore &store) const override;

private:
  llvm::SmallVector<std::unique_ptr<Constraint<Item>>> constrs;
};

//===----------------------------------------------------------------------===//
// Always true constraint
//===----------------------------------------------------------------------===//

/// Always true constraint.
/// All types satisfy this constraint.
template <class Item>
class AnyConstraint : public Constraint<Item> {
public:
  AnyConstraint() {}

  virtual LogicalResult
  verify(Optional<function_ref<InFlightDiagnostic()>> emitError, Item item,
         VarConstraints const &cstrs, VarStore &store) const override {
    return success();
  };
};

//===----------------------------------------------------------------------===//
// Variable constraint
//===----------------------------------------------------------------------===//

/// Constraint variable.
/// All items matching the variable should be equal. The first item
/// matching the variable is the one setting the value.
template <class Item>
class VarConstraint : public Constraint<Item> {
public:
  VarConstraint(size_t varIndex) : varIndex{varIndex} {}

  virtual LogicalResult
  verify(Optional<function_ref<InFlightDiagnostic()>> emitError, Item item,
         VarConstraints const &cstrs, VarStore &store) const override;

private:
  size_t varIndex;
};

//===----------------------------------------------------------------------===//
// Base constraint
//===----------------------------------------------------------------------===//

/// Type constraint asserting that the base item is of a certain dynamic item.
class DynTypeBaseConstraint : public Constraint<Type> {
public:
  DynTypeBaseConstraint(DynamicTypeDefinition *dynTypeDef)
      : dynTypeDef(dynTypeDef) {}

  virtual LogicalResult
  verify(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
         VarConstraints const &cstrs, VarStore &store) const override;

private:
  DynamicTypeDefinition *dynTypeDef;
};

/// Type constraint asserting that the base type is of a certain C++-defined
/// type.
class TypeBaseConstraint : public Constraint<Type> {
public:
  TypeBaseConstraint(TypeWrapper *typeDef) : typeDef(typeDef) {}

  virtual LogicalResult
  verify(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
         VarConstraints const &cstrs, VarStore &store) const override;

private:
  /// Base type that satisfies the constraint.
  TypeWrapper *typeDef;
};

//===----------------------------------------------------------------------===//
// Parameters constraint
//===----------------------------------------------------------------------===//

/// Type constraint having constraints on dynamic type parameters.
/// A type satisfies this constraint if it has the right expected type,
/// and if each of its parameter satisfies their associated constraint.
class DynTypeParamsConstraint : public Constraint<Type> {
public:
  DynTypeParamsConstraint(
      DynamicTypeDefinition *dynTypeDef,
      llvm::SmallVector<std::unique_ptr<Constraint<Type>>>
          &&paramConstraints)
      : dynTypeDef(dynTypeDef), paramConstraints(std::move(paramConstraints)) {}

  virtual LogicalResult
  verify(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
         VarConstraints const &cstrs, VarStore &store) const override;

private:
  /// TypeID of the parametric type that satisfies this constraint.
  DynamicTypeDefinition *dynTypeDef;

  /// Type constraints of the type parameters.
  llvm::SmallVector<std::unique_ptr<Constraint<Type>>> paramConstraints;
};

/// Type constraint having constraints on C++-defined type parameters.
/// A type satisfies this constraint if it has the right expected type,
/// and if each of its parameter satisfies their associated constraint.
class TypeParamsConstraint : public Constraint<Type> {
public:
  TypeParamsConstraint(TypeWrapper *typeDef,
                       llvm::SmallVector<std::unique_ptr<Constraint<Type>>>
                           &&paramConstraints)
      : typeDef(typeDef), paramConstraints(std::move(paramConstraints)) {}

  virtual LogicalResult
  verify(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
         VarConstraints const &cstrs, VarStore &store) const override;

private:
  /// Base type that satisfies the constraint.
  TypeWrapper *typeDef;

  /// Type constraints of the type parameters.
  llvm::SmallVector<std::unique_ptr<Constraint<Type>>> paramConstraints;
};

} // namespace irdl
} // namespace mlir

#endif // MLIR_DIALECT_IRDL_IR_IRDLCONSTRAINT_H_
