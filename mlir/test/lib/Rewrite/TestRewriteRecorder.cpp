#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/RewriteListener.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

struct StrengthReduceMultiply : OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhs().getType() != op.getRhs().getType())
      return failure();

    auto rhs = op.getRhs();
    if (auto constOp = llvm::dyn_cast<arith::ConstantOp>(rhs.getDefiningOp())) {
      auto integer = cast<IntegerAttr>(constOp.getValue()).getInt();
      if (integer == 2) {
        rewriter.replaceOpWithNewOp<arith::AddIOp>(op, op.getLhs(),
                                                   op.getLhs());
        return success();
      }
    }

    return failure();
  }
};

struct TestRewriteRecord : PassWrapper<TestRewriteRecord, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRewriteRecord)
  StringRef getArgument() const override { return "test-rewrite-record"; }
  StringRef getDescription() const override {
    return "Test the RewriteRecord functionality";
  }

  void runOnOperation() override {
    PassRewriteRecorder recorder;
    GreedyRewriteConfig config;
    config.listener = &recorder;
    auto ctx = &getContext();

    RewritePatternSet set{ctx};
    set.add<StrengthReduceMultiply>(ctx);

    (void)applyPatternsGreedily(getOperation(), std::move(set), config);

    for (auto &record : recorder.getRecords()) {
      for (auto &added : record.added)
        llvm::errs() << "add: " << added << "\n";
      for (auto &added : record.removed)
        llvm::errs() << "rem: " << added << "\n";
    }

    for (auto &elem : recorder.dce)
      llvm::errs() << "rem by dce: " << elem << " \n";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

namespace mlir::test {
void registerTestRewriteRecordPass() { PassRegistration<TestRewriteRecord>(); }
} // namespace mlir::test