//===- RuntimeOpVerification.cpp - Op Verification ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Operation.h"

namespace mlir {
#define GEN_PASS_DEF_KNOWNBITSANALYSIS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct KnownBitsAnalysisPass
    : public impl::KnownBitsAnalysisBase<
          KnownBitsAnalysisPass> {
  void runOnOperation() override;
};
} // namespace

void KnownBitsAnalysisPass::runOnOperation() {
  llvm::errs() << "Ran known bits analysis.\n";
}

std::unique_ptr<Pass> mlir::createKnownBitsAnalysisPass() {
  return std::make_unique<KnownBitsAnalysisPass>();
}

