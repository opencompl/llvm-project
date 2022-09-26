//===- mlir-irdl-opt.cpp - MLIR Optimizer Driver --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-irdl-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace irdl;

class ComplexTypeWrapper : public CppTypeWrapper<ComplexType> {
  StringRef getName() override { return "builtin.complex"; }

  SmallVector<Attribute> getParameters(ComplexType type) override {
    return {TypeAttr::get(type.getElementType())};
  }

  size_t getParameterAmount() override { return 1; }

  Type instantiate(llvm::function_ref<InFlightDiagnostic()> emitError,
                   ArrayRef<Attribute> parameters) override {
    if (parameters.size() != this->getParameterAmount()) {
      emitError().append("invalid number of type parameters ",
                         parameters.size(), " (expected ",
                         this->getParameterAmount(), ")");
      return Type();
    }

    return ComplexType::getChecked(emitError,
                                   parameters[0].cast<TypeAttr>().getValue());
  }
};

int main(int argc, char **argv) {
  registerAllPasses();
  DialectRegistry registry;
  registerAllDialects(registry);

  MLIRContext context(registry, MLIRContext::Threading::DISABLED);

  // Register wrappers around C++ types and attributes.
  auto irdl = context.getOrLoadDialect<irdl::IRDLDialect>();
  irdl->addTypeWrapper<ComplexTypeWrapper>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry,
                        /*preloadDialectsInContext=*/false, &context));
}
