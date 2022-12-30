//===- IRDLAttributes.h - Attributes definition for IRDL --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes used in the IRDL dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDLAttributes.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "mlir/Dialect/IRDL/IRDLContext.h"
#include "mlir/Dialect/IRDL/TypeWrapper.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Metadata.h"

//===----------------------------------------------------------------------===//
// TypeWrapper parameter
//===----------------------------------------------------------------------===//

using namespace mlir;
using namespace irdl;

namespace mlir {
AsmPrinter &operator<<(AsmPrinter &printer, TypeWrapper *param) {
  printer << param->getName();
  return printer;
}

template <>
struct FieldParser<TypeWrapper *> {
  static FailureOr<TypeWrapper *> parse(AsmParser &parser) {
    std::string name;
    (void)parser.parseOptionalKeywordOrString(&name);
    auto *irdl = parser.getContext()->getOrLoadDialect<IRDLDialect>();
    auto typeWrapper = irdl->getTypeWrapper(name);
    if (!typeWrapper)
      return parser.emitError(parser.getCurrentLocation(), "Type wrapper ")
             << name << " was not registered in IRDL";
    return typeWrapper;
  }
};
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"

namespace mlir {
namespace irdl {

void IRDLDialect::registerAttributes() {
#define GET_ATTRDEF_LIST
  addAttributes<
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"
      >();
}

} // namespace irdl
} // namespace mlir

//===----------------------------------------------------------------------===//
// IRDL equality type constraint attribute
//===----------------------------------------------------------------------===//

std::unique_ptr<Constraint<Type>> EqTypeConstraintAttr::getTypeConstraint(
    IRDLContext &irdlCtx,
    llvm::SmallMapVector<StringRef, std::unique_ptr<Constraint<Type>>,
                         4> const &constrVars) const {
  return std::make_unique<EqConstraint<Type>>(getType());
}

//===----------------------------------------------------------------------===//
// Always true type constraint attribute
//===----------------------------------------------------------------------===//

std::unique_ptr<Constraint<Type>> AnyTypeConstraintAttr::getTypeConstraint(
    IRDLContext &irdlCtx,
    llvm::SmallMapVector<StringRef, std::unique_ptr<Constraint<Type>>,
                         4> const &constrVars) const {
  return std::make_unique<AnyConstraint<Type>>();
}

//===----------------------------------------------------------------------===//
// IRDL AnyOf type constraint attribute
//===----------------------------------------------------------------------===//

std::unique_ptr<Constraint<Type>>
AnyOfTypeConstraintAttr::getTypeConstraint(
    IRDLContext &irdlCtx,
    llvm::SmallMapVector<StringRef, std::unique_ptr<Constraint<Type>>,
                         4> const &constrVars) const {
  SmallVector<std::unique_ptr<Constraint<Type>>> constraints;
  auto constraintAttrs = getConstrs();
  for (auto constrAttr : constraintAttrs)
    constraints.push_back(
        constrAttr.cast<TypeConstraintAttrInterface>().getTypeConstraint(
            irdlCtx, constrVars));
  return std::make_unique<AnyOfConstraint<Type>>(std::move(constraints));
}

//===----------------------------------------------------------------------===//
// IRDL And type constraint attribute
//===----------------------------------------------------------------------===//

std::unique_ptr<Constraint<Type>> AndTypeConstraintAttr::getTypeConstraint(
    IRDLContext &irdlCtx,
    llvm::SmallMapVector<StringRef, std::unique_ptr<Constraint<Type>>,
                         4> const &constrVars) const {
  SmallVector<std::unique_ptr<Constraint<Type>>> constraints;
  auto constraintAttrs = getConstrs();
  for (auto constrAttr : constraintAttrs)
    constraints.push_back(
        constrAttr.cast<TypeConstraintAttrInterface>().getTypeConstraint(
            irdlCtx, constrVars));
  return std::make_unique<AndConstraint<Type>>(std::move(constraints));
}

//===----------------------------------------------------------------------===//
// Type constraint variable
//===----------------------------------------------------------------------===//

std::unique_ptr<Constraint<Type>> VarTypeConstraintAttr::getTypeConstraint(
    IRDLContext &irdlCtx,
    llvm::SmallMapVector<StringRef, std::unique_ptr<Constraint<Type>>,
                         4> const &constrVars) const {
  auto name = getName();
  // Iterate in reverse to match the latest defined variable.
  size_t i = 0;
  auto itr = constrVars.begin();
  for (; itr != constrVars.end(); itr++, i++) {
    if (itr->first == name) {
      return std::make_unique<VarConstraint<Type>>(i);
    }
  }
  // TODO: Make this an error
  assert(false && "Unknown type constraint variable");
}

//===----------------------------------------------------------------------===//
// Attribute for constraint on dynamic type base type
//===----------------------------------------------------------------------===/

// TODO: Replace with "findDynamicType" once it has been moved to
// a place suitable for use in IRDL.
DynamicTypeDefinition *resolveDynamicTypeDefinition(MLIRContext *ctx,
                                                    StringRef type) {
  auto splittedTypeName = type.split('.');
  auto dialectName = splittedTypeName.first;
  auto typeName = splittedTypeName.second;

  auto dialect = ctx->getOrLoadDialect(dialectName);
  assert(dialect && "dialect is not registered");
  auto extensibleDialect = llvm::dyn_cast<ExtensibleDialect>(dialect);
  assert(extensibleDialect && "dialect is not extensible");

  return extensibleDialect->lookupTypeDefinition(typeName);
}

std::unique_ptr<Constraint<Type>>
DynTypeBaseConstraintAttr::getTypeConstraint(
    IRDLContext &irdlCtx,
    llvm::SmallMapVector<StringRef, std::unique_ptr<Constraint<Type>>,
                         4> const &constrVars) const {
  auto *typeDef =
      resolveDynamicTypeDefinition(this->getContext(), this->getTypeName());

  return std::make_unique<DynTypeBaseConstraint>(typeDef);
}

//===----------------------------------------------------------------------===//
// Attribute for constraint on non-dynamic type base type
//===----------------------------------------------------------------------===/

std::unique_ptr<Constraint<Type>> TypeBaseConstraintAttr::getTypeConstraint(
    IRDLContext &irdlCtx,
    llvm::SmallMapVector<StringRef, std::unique_ptr<Constraint<Type>>,
                         4> const &constrVars) const {
  return std::make_unique<TypeBaseConstraint>(getTypeDef());
}

//===----------------------------------------------------------------------===//
// Attribute for constraint on dynamic type parameters
//===----------------------------------------------------------------------===//

std::unique_ptr<Constraint<Type>>
DynTypeParamsConstraintAttr::getTypeConstraint(
    IRDLContext &irdlCtx,
    llvm::SmallMapVector<StringRef, std::unique_ptr<Constraint<Type>>,
                         4> const &constrVars) const {
  SmallVector<std::unique_ptr<Constraint<Type>>> paramConstraints;
  for (auto paramConstraintAttr : getParamConstraints())
    paramConstraints.push_back(
        paramConstraintAttr.cast<TypeConstraintAttrInterface>()
            .getTypeConstraint(irdlCtx, constrVars));

  auto *typeDef =
      resolveDynamicTypeDefinition(this->getContext(), this->getTypeName());
  return std::make_unique<DynTypeParamsConstraint>(typeDef,
                                                   std::move(paramConstraints));
}

//===----------------------------------------------------------------------===//
// Attribute for constraint on non-dynamic type parameters
//===----------------------------------------------------------------------===//

std::unique_ptr<Constraint<Type>>
TypeParamsConstraintAttr::getTypeConstraint(
    IRDLContext &irdlCtx,
    llvm::SmallMapVector<StringRef, std::unique_ptr<Constraint<Type>>,
                         4> const &constrVars) const {
  SmallVector<std::unique_ptr<Constraint<Type>>> paramConstraints;
  for (auto paramConstraintAttr : getParamConstraints())
    paramConstraints.push_back(
        paramConstraintAttr.cast<TypeConstraintAttrInterface>()
            .getTypeConstraint(irdlCtx, constrVars));

  return std::make_unique<TypeParamsConstraint>(getTypeDef(),
                                                std::move(paramConstraints));
}
