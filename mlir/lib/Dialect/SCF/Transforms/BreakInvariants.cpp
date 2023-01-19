#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_BREAKVALUESCOPING
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace llvm;
using namespace mlir;
using scf::IfOp;
using scf::YieldOp;

namespace {

struct BreakValueScopingPattern : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (ifOp.getNumResults() == 0)
      return success();
    auto yield1 = llvm::cast<scf::YieldOp>(ifOp.thenBlock()->back());
    auto yield2 = llvm::cast<scf::YieldOp>(ifOp.elseBlock()->back());
    rewriter.setInsertionPointAfter(yield1);
    auto new_yield =
        rewriter.create<YieldOp>(yield1.getLoc(), yield2.getResults());
    rewriter.eraseOp(yield1);
    return success();
  }
};

struct BreakValueScoping
    : public impl::BreakValueScopingBase<BreakValueScoping> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<BreakValueScopingPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(parentOp, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createBreakValueScopingPass() {
  return std::make_unique<BreakValueScoping>();
}
