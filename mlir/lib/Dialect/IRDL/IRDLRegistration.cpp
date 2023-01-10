//===- IRDLRegistration.cpp - IRDL dialect registration ----------- C++ -*-===//
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

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"

using namespace mlir;
using namespace irdl;

namespace mlir {
namespace irdl {

namespace {
// Verifier used for dynamic types.
LogicalResult
irdlTypeVerifier(function_ref<InFlightDiagnostic()> emitError,
                 ArrayRef<Attribute> params,
                 ArrayRef<std::unique_ptr<TypeConstraint>> constraints,
                 ArrayRef<size_t> paramConstraints) {
  if (params.size() != paramConstraints.size()) {
    emitError().append("expected ", paramConstraints.size(),
                       " type arguments, but had ", params.size());
    return failure();
  }

  ConstraintVerifier verifier(constraints);

  for (size_t i = 0; i < params.size(); i++) {
    if (!params[i].isa<TypeAttr>()) {
      emitError().append(
          "only type attribute type parameters are currently supported");
      return failure();
    }

    if (failed(verifier.verifyType(emitError,
                                   params[i].cast<TypeAttr>().getValue(),
                                   paramConstraints[i]))) {
      return failure();
    }
  }
  return success();
}
} // namespace

} // namespace irdl
} // namespace mlir

namespace {

LogicalResult verifyOpDefConstraints(
    Operation *op, ArrayRef<std::unique_ptr<TypeConstraint>> constraints,
    ArrayRef<size_t> operandConstrs, ArrayRef<size_t> resultConstrs) {
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

  ConstraintVerifier verifier(constraints);

  /// Check that all operands satisfy the constraints.
  for (unsigned i = 0; i < numOperands; ++i) {
    auto operandType = op->getOperand(i).getType();
    if (failed(
            verifier.verifyType({emitError}, operandType, operandConstrs[i]))) {
      return failure();
    }
  }

  /// Check that all results satisfy the constraints.
  for (unsigned i = 0; i < numResults; ++i) {
    auto resultType = op->getResult(i).getType();
    if (failed(
            verifier.verifyType({emitError}, resultType, resultConstrs[i]))) {
      return failure();
    }
  }

  return success();
}
} // namespace

namespace mlir {
namespace irdl {
/// Register an operation represented by a `irdl.operation` operation.
WalkResult registerOperation(ExtensibleDialect *dialect, OperationOp op) {
  // Resolve SSA values to verifier constraint slots
  SmallVector<Value> constrToValue;
  for (auto &op : op->getRegion(0).getOps()) {
    if (llvm::isa<VerifyConstraintInterface>(op)) {
      assert(op.getNumResults() == 1 &&
             "IRDL constraint operations must have exactly one result");
      constrToValue.push_back(op.getResult(0));
    }
  }

  // Build the verifiers for each constraint slot
  SmallVector<std::unique_ptr<TypeConstraint>> constraints;
  for (Value v : constrToValue) {
    VerifyConstraintInterface op =
        llvm::cast<VerifyConstraintInterface>(v.getDefiningOp());
    auto verifier = op.getVerifier(constrToValue);
    if (!verifier.has_value()) {
      return WalkResult::interrupt();
    }
    constraints.push_back(std::move(*verifier));
  }

  SmallVector<size_t> operandConstraints;
  SmallVector<size_t> resultConstraints;

  // Gather which constraint slots correspond to operand constraints
  auto operandsOp = op.getOp<OperandsOp>();
  if (operandsOp.has_value()) {
    operandConstraints.reserve(operandsOp->getArgs().size());
    for (auto operand : operandsOp->getArgs()) {
      for (size_t i = 0; i < constrToValue.size(); i++) {
        if (constrToValue[i] == operand) {
          operandConstraints.push_back(i);
          break;
        }
      }
    }
  }

  // Gather which constraint slots correspond to result constraints
  auto resultsOp = op.getOp<ResultsOp>();
  if (resultsOp.has_value()) {
    resultConstraints.reserve(resultsOp->getArgs().size());
    for (auto result : resultsOp->getArgs()) {
      for (size_t i = 0; i < constrToValue.size(); i++) {
        if (constrToValue[i] == result) {
          resultConstraints.push_back(i);
          break;
        }
      }
    }
  }

  auto parser = [](OpAsmParser &parser, OperationState &result) {
    return failure();
  };
  auto printer = [](Operation *op, OpAsmPrinter &printer, StringRef) {
    printer.printGenericOp(op);
  };

  auto verifier =
      [constraints{std::move(constraints)},
       operandConstraints{std::move(operandConstraints)},
       resultConstraints{std::move(resultConstraints)}](Operation *op) {
        return verifyOpDefConstraints(op, constraints, operandConstraints,
                                      resultConstraints);
      };

  auto regionVerifier = [](Operation *op) { return success(); };

  auto opDef = DynamicOpDefinition::get(
      op.getName(), dialect, std::move(verifier), std::move(regionVerifier),
      std::move(parser), std::move(printer));
  dialect->registerDynamicOp(std::move(opDef));

  return WalkResult::advance();
}
} // namespace irdl
} // namespace mlir

static WalkResult registerType(ExtensibleDialect *dialect, TypeOp op) {
  // Resolve SSA values to verifier constraint slots
  SmallVector<Value> constrToValue;
  for (auto &op : op->getRegion(0).getOps()) {
    if (llvm::isa<VerifyConstraintInterface>(op)) {
      assert(op.getNumResults() == 1 &&
             "IRDL constraint operations must have exactly one result");
      constrToValue.push_back(op.getResult(0));
    }
  }

  // Build the verifiers for each constraint slot
  SmallVector<std::unique_ptr<TypeConstraint>> constraints;
  for (Value v : constrToValue) {
    VerifyConstraintInterface op =
        llvm::cast<VerifyConstraintInterface>(v.getDefiningOp());
    auto verifier = op.getVerifier(constrToValue);
    if (!verifier.has_value()) {
      return WalkResult::interrupt();
    }
    constraints.push_back(std::move(*verifier));
  }

  // Gather which constraint slots correspond to parameter constraints
  auto params = op.getOp<ParametersOp>();
  SmallVector<size_t> paramConstraints;
  if (params.has_value()) {
    paramConstraints.reserve(params->getArgs().size());
    for (auto param : params->getArgs()) {
      for (size_t i = 0; i < constrToValue.size(); i++) {
        if (constrToValue[i] == param) {
          paramConstraints.push_back(i);
          break;
        }
      }
    }
  }

  auto verifier = [paramConstraints{std::move(paramConstraints)},
                   constraints{std::move(constraints)}](
                      function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<Attribute> params) {
    return irdlTypeVerifier(emitError, params, constraints, paramConstraints);
  };

  auto type =
      DynamicTypeDefinition::get(op.getName(), dialect, std::move(verifier));

  dialect->registerDynamicType(std::move(type));

  return WalkResult::advance();
}

static WalkResult registerDialect(DialectOp op) {
  auto *ctx = op.getContext();
  auto dialectName = op.getName();

  ctx->getOrLoadDynamicDialect(dialectName, [](DynamicDialect *dialect) {});

  auto *dialect =
      llvm::dyn_cast<ExtensibleDialect>(ctx->getLoadedDialect(dialectName));
  assert(dialect && "extensible dialect should have been registered.");

  WalkResult res =
      op.walk([&](TypeOp op) { return registerType(dialect, op); });
  if (res.wasInterrupted())
    return res;

  return op.walk(
      [&](OperationOp op) { return registerOperation(dialect, op); });
}

namespace mlir {
namespace irdl {
LogicalResult registerDialects(ModuleOp op) {
  WalkResult res =
      op.walk([&](DialectOp dialect) { return registerDialect(dialect); });
  return failure(res.wasInterrupted());
}
} // namespace irdl
} // namespace mlir
