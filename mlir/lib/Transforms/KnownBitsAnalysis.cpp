//===- RuntimeOpVerification.cpp - Op Verification ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_KNOWNBITSANALYSIS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct ConstantKnownBitsPattern
    : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                PatternRewriter &rewriter) const override {
    auto ctx = constantOp.getContext();

    auto value = llvm::dyn_cast<IntegerAttr>(constantOp.getValueAttr());
    if (!value)
        return success();
    
    if (value.getType() != IntegerType::get(ctx, 32))
        return success();

    if (constantOp->getAttr("analysis"))
        return success();

    std::string analysis;
    for (int i = 0; i < 32; i++) {
        analysis += std::to_string((int)value.getValue()[31 - i]);
    }
    auto analysisAttr = StringAttr::get(ctx, analysis);

    rewriter.startRootUpdate(constantOp);
    constantOp->setAttr("analysis", analysisAttr);
    rewriter.finalizeRootUpdate(constantOp);
    return success();
  }
};

auto ANALYSIS_ATTR_NAME = "analysis";

StringAttr getAnalysis(Value val) {
    if (auto opRes = val.dyn_cast<OpResult>()) {
        auto owner = opRes.getOwner();
        if (auto analysis = owner->getAttr(ANALYSIS_ATTR_NAME)) {
            if (auto analysisStr = analysis.dyn_cast<StringAttr>())
            return analysisStr;
        }
    }
    return StringAttr::get(val.getContext(), "????????????????????????????????");
}

StringAttr join(StringAttr lhs, StringAttr rhs) {
    std::string analysis;
    auto lhsVal = lhs.getValue();
    auto rhsVal = rhs.getValue();
    for (int i = 0; i < 32; i++) {
        char c = '?';
        if (lhsVal[i] == rhsVal[i]) {
            c = lhsVal[i];
        }
        analysis += c;
    }
    return StringAttr::get(lhs.getContext(), analysis);
}

struct OrKnownBitsPattern
    : public OpRewritePattern<arith::OrIOp> {
  using OpRewritePattern<arith::OrIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::OrIOp OrOp,
                                PatternRewriter &rewriter) const override {
    auto ctx = OrOp.getContext();
    
    if (OrOp.getType() != IntegerType::get(ctx, 32))
        return success();

    auto analysisLhs = getAnalysis(OrOp.getLhs()).getValue();
    auto analysisRhs = getAnalysis(OrOp.getRhs()).getValue();

    std::string analysis;
    for (int i = 0; i < 32; i++) {
        char c = '?';
        if (analysisLhs[i] == '1' || analysisRhs[i] == '1') {
            c = '1';
        } else if (analysisLhs[i] == '0' && analysisRhs[i] == '0') {
            c = '0';
        }
        analysis += c;
    }
    auto analysisAttr = StringAttr::get(ctx, analysis);

    if (OrOp->getAttr(ANALYSIS_ATTR_NAME) == analysisAttr)
        return success();

    rewriter.startRootUpdate(OrOp);
    OrOp->setAttr(ANALYSIS_ATTR_NAME, analysisAttr);
    rewriter.finalizeRootUpdate(OrOp);
    return success();
  }
};
} // namespace


namespace {
struct KnownBitsAnalysisPass
    : public impl::KnownBitsAnalysisBase<
          KnownBitsAnalysisPass> {
  void runOnOperation() override;
};
} // namespace

void KnownBitsAnalysisPass::runOnOperation() {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ConstantKnownBitsPattern, OrKnownBitsPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(parentOp, std::move(patterns));

}

std::unique_ptr<Pass> mlir::createKnownBitsAnalysisPass() {
  return std::make_unique<KnownBitsAnalysisPass>();
}