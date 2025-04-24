//===- TestConstantFold.cpp - Pass to test constant folding ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;

namespace {

struct RewriteMul2ToAdd : OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    auto *rhsOp = op.getRhs().getDefiningOp();
    if (!rhsOp)
      return failure();

    if (auto constOp = llvm::dyn_cast<arith::ConstantOp>(rhsOp)) {
      if (auto val = dyn_cast<IntegerAttr>(constOp.getValueAttr())) {
        if (val.getInt() == 2) {
          auto fusedLoc = rewriter.getFusedLoc({constOp.getLoc(), op.getLoc()});
          auto add = rewriter.create<arith::AddIOp>(fusedLoc, op.getLhs(),
                                                    op.getLhs());
          rewriter.replaceOp(op, add);
          return success();
        }
      }
    }
    return failure();
  }
};
/// Simple constant folding pass.
struct TestConstantMulToAdd
    : public PassWrapper<TestConstantMulToAdd, OperationPass<>>,
      public RewriterBase::Listener {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestConstantMulToAdd)

  StringRef getArgument() const final { return "test-constant-mul"; }
  StringRef getDescription() const final {
    return "Test operation constant folding";
  }

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace

void TestConstantMulToAdd::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patternSet{ctx};
  patternSet.add<RewriteMul2ToAdd>(ctx);
  (void)applyPatternsGreedily(getOperation(), std::move(patternSet));
}

namespace mlir {
namespace test {
void registerTestConstantMulToAdd() {
  PassRegistration<TestConstantMulToAdd>();
}
} // namespace test
} // namespace mlir
