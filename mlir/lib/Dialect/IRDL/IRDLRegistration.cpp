//===- IRDLRegistration.cpp - IRDL registration -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration of MLIR objects from IRDL operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IRDLRegistration.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.h"
#include "mlir/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "mlir/Dialect/IRDL/IRDLConstraint.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"

using namespace mlir;
using namespace irdl;

namespace mlir {
namespace irdl {

namespace {
// Verifier used for dynamic types.
LogicalResult irdlTypeVerifier(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<Attribute> params,
    ArrayRef<std::unique_ptr<IRDLConstraint<Type>>> constraintVars,
    ArrayRef<std::unique_ptr<IRDLConstraint<Type>>> paramConstraints) {
  if (params.size() != paramConstraints.size()) {
    emitError().append("expected ", paramConstraints.size(),
                       " type arguments, but had ", params.size());
    return failure();
  }

  VarStore store(constraintVars.size(), 0);
  VarConstraints cstrts(constraintVars, {});

  for (size_t i = 0; i < params.size(); i++) {
    if (failed(paramConstraints[i]->verify(
            emitError, params[i].cast<TypeAttr>().getValue(), cstrts, store)))
      return failure();
  }
  return success();
}
} // namespace

} // namespace irdl
} // namespace mlir

namespace {

LogicalResult verifyOpDefConstraints(
    Operation *op,
    ArrayRef<std::unique_ptr<IRDLConstraint<Type>>> constraintVars,
    ArrayRef<std::unique_ptr<IRDLConstraint<Type>>> operandConstrs,
    ArrayRef<std::unique_ptr<IRDLConstraint<Type>>> resultConstrs) {
  /// Check that we have the right number of operands.
  auto numOperands = op->getNumOperands();
  auto numExpectedOperands = operandConstrs.size();
  if (numOperands != numExpectedOperands)
    return op->emitOpError(std::to_string(numExpectedOperands) +
                           " operands expected, but got " +
                           std::to_string(numOperands));

  /// Check that we have the right number of results.
  auto numResults = op->getNumResults();
  auto numExpectedResults = resultConstrs.size();
  if (numResults != numExpectedResults)
    return op->emitOpError(std::to_string(numExpectedResults) +
                           " results expected, but got " +
                           std::to_string(numResults));

  auto emitError = [op]() { return op->emitError(); };
  VarStore store(constraintVars.size(), 0);
  VarConstraints cstrts(constraintVars, {});

  /// Check that all operands satisfy the constraints.
  for (unsigned i = 0; i < numOperands; ++i) {
    auto operandType = op->getOperand(i).getType();
    auto &constraint = operandConstrs[i];
    if (failed(constraint->verify({emitError}, operandType, cstrts, store)))
      return failure();
  }

  /// Check that all results satisfy the constraints.
  for (unsigned i = 0; i < numResults; ++i) {
    auto resultType = op->getResult(i).getType();
    auto &constraint = resultConstrs[i];
    if (failed(constraint->verify({emitError}, resultType, cstrts, store)))
      return failure();
  }

  return success();
}
} // namespace

namespace mlir {
namespace irdl {
/// Register an operation represented by a `irdl.operation` operation.
void registerOperation(IRDLContext &irdlCtx, ExtensibleDialect *dialect,
                       OperationOp op) {
  llvm::SmallMapVector<StringRef, std::unique_ptr<IRDLConstraint<Type>>, 4>
      constraintVars;
  SmallVector<std::unique_ptr<IRDLConstraint<Type>>> operandConstraints;
  SmallVector<std::unique_ptr<IRDLConstraint<Type>>> resultConstraints;

  auto constraintVarsOp = op.getOp<ConstraintVarsOp>();
  if (constraintVarsOp) {
    for (auto constraint : constraintVarsOp->getParams().getValue()) {
      auto constraintAttr = constraint.cast<NamedTypeConstraintAttr>();
      auto constraintConstr = constraintAttr.getConstraint()
                                  .cast<TypeConstraintAttrInterface>()
                                  .getTypeConstraint(irdlCtx, constraintVars);

      // TODO: make this an error
      assert(constraintVars.count(constraintAttr.getName()) == 0 &&
             "Two variables share the same name");

      constraintVars.insert(std::make_pair(constraintAttr.getName(),
                                           std::move(constraintConstr)));
    }
  }

  // Add the operand constraints to the type constraints.
  auto operandsOp = op.getOp<OperandsOp>();
  if (operandsOp.has_value()) {
    operandConstraints.reserve(operandsOp->getParams().size());
    for (auto operand : operandsOp->getParams().getValue()) {
      auto operandAttr = operand.cast<NamedTypeConstraintAttr>();
      auto constraint = operandAttr.getConstraint()
                            .cast<TypeConstraintAttrInterface>()
                            .getTypeConstraint(irdlCtx, constraintVars);
      operandConstraints.emplace_back(std::move(constraint));
    }
  }

  // Add the result constraints to the type constraints.
  auto resultsOp = op.getOp<ResultsOp>();
  if (resultsOp.has_value()) {
    resultConstraints.reserve(resultsOp->getParams().size());
    for (auto result : resultsOp->getParams().getValue()) {
      auto resultAttr = result.cast<NamedTypeConstraintAttr>();
      auto constraint = resultAttr.getConstraint()
                            .cast<TypeConstraintAttrInterface>()
                            .getTypeConstraint(irdlCtx, constraintVars);
      resultConstraints.emplace_back(std::move(constraint));
    }
  }

  auto parser = [](OpAsmParser &parser, OperationState &result) {
    return failure();
  };
  auto printer = [](Operation *op, OpAsmPrinter &printer, StringRef) {
    printer.printGenericOp(op);
  };

  SmallVector<std::unique_ptr<IRDLConstraint<Type>>> constraintVarsConstrs;
  for (auto &constrVar : constraintVars) {
    constraintVarsConstrs.emplace_back(std::move(constrVar.second));
  }

  auto verifier =
      [constraintVarsConstrs{std::move(constraintVarsConstrs)},
       operandConstraints{std::move(operandConstraints)},
       resultConstraints{std::move(resultConstraints)}](Operation *op) {
        return verifyOpDefConstraints(op, constraintVarsConstrs,
                                      operandConstraints, resultConstraints);
      };

  auto regionVerifier = [](Operation *op) { return success(); };

  auto opDef = DynamicOpDefinition::get(
      op.getName(), dialect, std::move(verifier), std::move(regionVerifier),
      std::move(parser), std::move(printer));
  dialect->registerDynamicOp(std::move(opDef));
}
} // namespace irdl
} // namespace mlir

static void registerType(IRDLContext &irdlCtx, ExtensibleDialect *dialect,
                         TypeOp op) {
  auto params = op.getOp<ParametersOp>();

  SmallVector<std::unique_ptr<IRDLConstraint<Type>>> paramConstraints;
  if (params.has_value()) {
    for (auto param : params->getParams().getValue()) {
      paramConstraints.push_back(param.cast<NamedTypeConstraintAttr>()
                                     .getConstraint()
                                     .cast<TypeConstraintAttrInterface>()
                                     .getTypeConstraint(irdlCtx, {}));
    }
  }

  llvm::SmallMapVector<StringRef, std::unique_ptr<IRDLConstraint<Type>>, 4>
      constraintVars;
  auto constraintVarsOp = op.getOp<ConstraintVarsOp>();
  if (constraintVarsOp) {
    for (auto constraint : constraintVarsOp->getParams().getValue()) {
      auto constraintAttr = constraint.cast<NamedTypeConstraintAttr>();
      auto constraintConstr = constraintAttr.getConstraint()
                                  .cast<TypeConstraintAttrInterface>()
                                  .getTypeConstraint(irdlCtx, constraintVars);

      // TODO: make this an error
      assert(constraintVars.count(constraintAttr.getName()) == 0 &&
             "Two variables share the same name");

      llvm::errs() << "registered variable " << constraintAttr.getName()
                   << ", current has "
                   << constraintVars.count(constraintAttr.getName())
                   << "registered.\n";

      constraintVars.insert(std::make_pair(constraintAttr.getName(),
                                           std::move(constraintConstr)));
    }
  }

  SmallVector<std::unique_ptr<IRDLConstraint<Type>>> constraintVarsConstrs;
  for (auto &constrVar : constraintVars) {
    constraintVarsConstrs.emplace_back(std::move(constrVar.second));
  }

  auto verifier = [constraintVarsConstrs{std::move(constraintVarsConstrs)},
                   paramConstraints{std::move(paramConstraints)}](
                      function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<Attribute> params) {
    return irdlTypeVerifier(emitError, params, constraintVarsConstrs,
                            paramConstraints);
  };

  auto type =
      DynamicTypeDefinition::get(op.getName(), dialect, std::move(verifier));

  dialect->registerDynamicType(std::move(type));
}

static void registerDialect(IRDLContext &irdlCtx, DialectOp op) {
  auto *ctx = op.getContext();
  auto dialectName = op.getName();

  ctx->getOrLoadDynamicDialect(dialectName, [](DynamicDialect *dialect) {});

  auto *dialect =
      llvm::dyn_cast<ExtensibleDialect>(ctx->getLoadedDialect(dialectName));
  assert(dialect && "extensible dialect should have been registered.");

  op.walk([&](TypeOp op) { registerType(irdlCtx, dialect, op); });
  op.walk([&](OperationOp op) { registerOperation(irdlCtx, dialect, op); });
}

namespace mlir {
namespace irdl {
void registerDialects(ModuleOp op) {
  IRDLContext &irdlCtx =
      llvm::cast<IRDLDialect>(op.getContext()->getOrLoadDialect("irdl"))
          ->irdlContext;
  op.walk([&](DialectOp dialect) { registerDialect(irdlCtx, dialect); });
}
} // namespace irdl
} // namespace mlir
