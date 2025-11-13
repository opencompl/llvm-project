//===- OpDefinitionsGen.cpp - IRDL op definitions generator ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpDefinitionsGen uses the description of operations to generate IRDL
// definitions for ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include <chrono>

using namespace llvm;
using namespace mlir;

namespace pattern {

struct AddConstantFolding : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp addOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    auto cst0 = lhs.getDefiningOp<arith::ConstantIntOp>();
    if (!cst0)
      return failure();

    auto cst1 = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!cst1)
      return failure();

    auto foldCst = rewriter.create<arith::ConstantOp>(
        addOp.getLoc(),
        IntegerAttr::get(IntegerType::get(addOp.getContext(), 32),
                         cst0.value() + cst1.value()));

    // Replace add with the folded constant result.
    rewriter.replaceOp(addOp, foldCst.getResult());

    // Clean up now-dead constants if they became unused.
    if (lhs.use_empty())
      rewriter.eraseOp(cst0);
    if (rhs.use_empty())
      rewriter.eraseOp(cst1);

    return success();
  }
};

struct AddZeroFolding : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::AddIOp addOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    auto cst = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!cst)
      return failure();

    if (cst.value() != 0)
      return failure();

    rewriter.replaceOp(addOp, lhs);
    if (rhs.getUses().empty())
      rewriter.eraseOp(rhs.getDefiningOp());
    return success();
  }
};

struct MulTwoReduce : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MulIOp mulOp,
                                PatternRewriter &rewriter) const override {
    auto lhs = mulOp.getLhs();
    auto rhs = mulOp.getRhs();

    auto cstRhs = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!cstRhs)
      return failure();

    if (cstRhs.value() != 2)
      return failure();

    auto addOp = rewriter.create<arith::AddIOp>(mulOp.getLoc(), lhs, lhs);

    // Replace add with the folded constant result.
    rewriter.replaceOp(mulOp, addOp.getResult());

    // Clean up now-dead constants if they became unused.
    if (rhs.use_empty())
      rewriter.eraseOp(cstRhs);

    return success();
  }
};

} // namespace pattern

namespace custom {

static Operation *addConstantFolding(Operation *op) {
  auto addOp = dyn_cast<arith::AddIOp>(op);
  if (!addOp)
    return nullptr;
  auto lhs = addOp.getLhs();
  auto rhs = addOp.getRhs();

  auto cst0 = lhs.getDefiningOp<arith::ConstantIntOp>();
  if (!cst0)
    return nullptr;

  auto cst1 = rhs.getDefiningOp<arith::ConstantIntOp>();
  if (!cst1)
    return nullptr;

  OpBuilder builder(op);
  auto foldCst = builder.create<arith::ConstantOp>(
      UnknownLoc::get(op->getContext()),
      IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                       cst0.value() + cst1.value()));

  addOp.replaceAllUsesWith(foldCst.getResult());
  addOp.erase();
  if (lhs.getUses().empty())
    lhs.getDefiningOp()->erase();
  if (rhs.getUses().empty())
    rhs.getDefiningOp()->erase();

  return foldCst;
}

static Operation *addZeroFolding(Operation *op) {
  auto addOp = dyn_cast<arith::AddIOp>(op);
  if (!addOp)
    return nullptr;
  auto lhs = addOp.getLhs();
  auto rhs = addOp.getRhs();

  auto cst0 = rhs.getDefiningOp<arith::ConstantIntOp>();
  if (!cst0)
    return nullptr;

  if (cst0.value() != 0)
    return nullptr;

  addOp.replaceAllUsesWith(lhs);
  addOp.erase();

  if (rhs.getUses().empty())
    rhs.getDefiningOp()->erase();

  return nullptr;
}

static Operation *mulTwoReduce(Operation *op) {
  auto mulOp = dyn_cast<arith::MulIOp>(op);
  if (!mulOp)
    return nullptr;
  auto lhs = mulOp.getLhs();
  auto rhs = mulOp.getRhs();

  auto cstRhs = rhs.getDefiningOp<arith::ConstantIntOp>();
  if (!cstRhs)
    return nullptr;

  if (cstRhs.value() != 2)
    return nullptr;

  OpBuilder builder(op);
  auto addOp = builder.create<arith::AddIOp>(mulOp.getLoc(), lhs, lhs);

  mulOp.replaceAllUsesWith(addOp.getResult());
  mulOp.erase();

  if (rhs.getUses().empty())
    cstRhs.erase();

  return nullptr;
}

} // namespace custom

template <typename Op, Operation *Rewrite(Operation *)>
static void rewriteFirst(ModuleOp module) {
  Operation *op = &module.getBody()->getOperations().front();
  while (op && !dyn_cast<Op>(op)) {
    op = op->getNextNode();
  }

  Rewrite(op);
}

template <Operation *Rewrite(Operation *)>
static void rewriteForwards(ModuleOp module) {
  Operation *op = &module.getBody()->getOperations().front();
  while (op) {
    Operation *next = op->getNextNode();
    auto *newOp = Rewrite(op);
    (void)newOp;
    // if (newOp) {
    //   op = newOp;
    //   continue;
    // }
    op = next;
  }
}

template <typename Pattern>
static void rewriteWorklist(ModuleOp module) {
  auto *ctx = module->getContext();

  mlir::GreedyRewriteConfig config;
  config.cseConstants = false;
  config.fold = false;
  config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
  config.useTopDownTraversal = true;

  mlir::RewritePatternSet patterns(ctx);
  patterns.insert<Pattern>(ctx);
  (void)mlir::applyPatternsGreedily(module, std::move(patterns), config);
}

namespace program {

// Create a program that looks like:
// func @main() -> i32 {
//   %0 = arith.constant [root] : i32
//   %1 = arith.constant [inc] : i32
//   %2 = [opcode] %0, %1 : i32
//   %3 = arith.constant [inc] : i32
//   %4 = [opcode] %2, %3 : i32
//   ...
template <typename Op>
static OwningOpRef<ModuleOp> constFoldTree(MLIRContext &ctx, uint64_t size,
                                           uint32_t root, uint32_t increment) {
  OpBuilder builder(&ctx);
  OwningOpRef<ModuleOp> module =
      builder.create<ModuleOp>(UnknownLoc::get(&ctx));
  builder = builder.atBlockBegin(module->getBody());

  Operation *accOp = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx),
      IntegerAttr::get(IntegerType::get(&ctx, 32), root));

  for (uint64_t i = 0; i < size; ++i) {
    auto cst = builder.create<arith::ConstantOp>(
        UnknownLoc::get(&ctx),
        IntegerAttr::get(IntegerType::get(&ctx, 32), increment));

    accOp = builder.create<Op>(UnknownLoc::get(&ctx), accOp->getResult(0),
                               cst->getResult(0));
  }

  OperationState state(UnknownLoc::get(&ctx), "test.test");
  state.addOperands(accOp->getResult(0));
  builder.create(state);

  return module;
}

static OwningOpRef<ModuleOp> addZeroTree(MLIRContext &ctx, uint64_t size) {
  return constFoldTree<arith::AddIOp>(ctx, size, 42, 0);
}

static OwningOpRef<ModuleOp> addOneTree(MLIRContext &ctx, uint64_t size) {
  return constFoldTree<arith::AddIOp>(ctx, size, 42, 1);
}

static OwningOpRef<ModuleOp> mulTwoTree(MLIRContext &ctx, uint64_t size) {
  return constFoldTree<arith::MulIOp>(ctx, size, 42, 2);
}

// Create a program that looks like:
// func @main() -> i32 {
//   %0 = arith.constant [root] : i32
//   %reuse = arith.constant [inc]: i32
//   %2 = [opcode] %0, %reuse : i32
//   %3 = [opcode] %2, %reuse : i32
//   ...
template <typename Op>
static OwningOpRef<ModuleOp> constReuseTree(MLIRContext &ctx, uint64_t size,
                                            uint32_t root, uint32_t increment) {
  OpBuilder builder(&ctx);
  OwningOpRef<ModuleOp> module =
      builder.create<ModuleOp>(UnknownLoc::get(&ctx));
  builder = builder.atBlockBegin(module->getBody());

  Operation *accOp = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx),
      IntegerAttr::get(IntegerType::get(&ctx, 32), root));

  Operation *reuseOp = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx),
      IntegerAttr::get(IntegerType::get(&ctx, 32), increment));

  for (uint64_t i = 0; i < size; ++i) {
    accOp = builder.create<Op>(UnknownLoc::get(&ctx), accOp->getResult(0),
                               reuseOp->getResult(0));
  }

  OperationState state(UnknownLoc::get(&ctx), "test.test");
  state.addOperands(accOp->getResult(0));
  builder.create(state);

  return module;
}

static OwningOpRef<ModuleOp> addZeroReuseTree(MLIRContext &ctx, uint64_t size) {
  return constReuseTree<arith::AddIOp>(ctx, size, 42, 0);
}

// Create a program that looks like:
// func @main() -> i32 {
//   %0 = arith.constant [lhs] : i32
//   %1 = arith.constant [rhs] : i32
//   %reuse = [opcode] %0, %1 : i32
//   %3 = [opcode] %reuse, %reuse : i32
//   %4 = [opcode] %3, %reuse : i32
//   %5 = [opcode] %4, %reuse : i32
//  ...
template <typename Op>
static OwningOpRef<ModuleOp> constLotsOfReuseTree(MLIRContext &ctx,
                                                  uint64_t size, uint32_t lhs,
                                                  uint32_t rhs) {
  OpBuilder builder(&ctx);
  OwningOpRef<ModuleOp> module =
      builder.create<ModuleOp>(UnknownLoc::get(&ctx));
  builder = builder.atBlockBegin(module->getBody());

  Operation *lhsOp = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx), IntegerAttr::get(IntegerType::get(&ctx, 32), lhs));

  Operation *rhsOp = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx), IntegerAttr::get(IntegerType::get(&ctx, 32), rhs));

  Operation *reuseOp = builder.create<Op>(
      UnknownLoc::get(&ctx), lhsOp->getResult(0), rhsOp->getResult(0));

  Operation *accOp = reuseOp;

  for (uint64_t i = 0; i < size; ++i) {
    accOp = builder.create<Op>(UnknownLoc::get(&ctx), accOp->getResult(0),
                               reuseOp->getResult(0));
  }

  OperationState state(UnknownLoc::get(&ctx), "test.test");
  state.addOperands(accOp->getResult(0));
  builder.create(state);

  return module;
}

static OwningOpRef<ModuleOp> addZeroLotsOfReuseTree(MLIRContext &ctx,
                                                    uint64_t size) {
  return constLotsOfReuseTree<arith::AddIOp>(ctx, size, 42, 0);
}

} // namespace program

template <typename F>
static auto time(std::string_view name, F f) {
  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();

  const auto print_time = [&]() {
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    llvm::errs() << name << " time (s): " << elapsed / std::chrono::seconds(1)
                 << "\n";
  };

  if constexpr (std::is_void_v<decltype(f())>) {
    f();
    print_time();
    return;
  } else {
    auto ret = f();
    print_time();
    return ret;
  }
}

template <typename Create, typename Rewrite>
static OwningOpRef<ModuleOp> run(MLIRContext &ctx, uint64_t size, Create create,
                                 Rewrite rewrite, bool print) {
  auto module = time("create", [&]() { return create(ctx, size); });
  time("rewrite", [&]() { return rewrite(*module); });

  if (print) {
    module->dump();
  }

  return module;
}

static OwningOpRef<ModuleOp> runBench(MLIRContext &ctx, std::string_view name,
                                      uint64_t size) {
  using namespace program;

  // clang-format off
  if (name == "add-fold-worklist")       { return run(ctx, size, addOneTree,       rewriteWorklist<pattern::AddConstantFolding>, true); }
  if (name == "add-zero-worklist")       { return run(ctx, size, addZeroTree,      rewriteWorklist<pattern::AddZeroFolding>,     true); }
  if (name == "add-zero-reuse-worklist") { return run(ctx, size, addZeroReuseTree, rewriteWorklist<pattern::AddZeroFolding>,     true); }
  if (name == "mul-two-worklist")        { return run(ctx, size, mulTwoTree,       rewriteWorklist<pattern::MulTwoReduce>,      false); }

  if (name == "add-fold-forwards")       { return run(ctx, size, addOneTree,       rewriteForwards<custom::addConstantFolding>,  true); }
  if (name == "add-zero-forwards")       { return run(ctx, size, addZeroTree,      rewriteForwards<custom::addZeroFolding>,      true); }
  if (name == "add-zero-reuse-forwards") { return run(ctx, size, addZeroReuseTree, rewriteForwards<custom::addZeroFolding>,      true); }
  if (name == "mul-two-forwards")        { return run(ctx, size, mulTwoTree,       rewriteForwards<custom::mulTwoReduce>,       false); }

  if (name == "add-zero-reuse-first")         { return run(ctx, size, addZeroReuseTree,       rewriteFirst<arith::AddIOp, custom::addZeroFolding>, false); }
  if (name == "add-zero-lots-of-reuse-first") { return run(ctx, size, addZeroLotsOfReuseTree, rewriteFirst<arith::AddIOp, custom::addZeroFolding>, false); }
  // clang-format on

  assert(false && "Unknown benchmark");
}

cl::opt<std::string> BenchmarkMode(cl::Positional, cl::desc("<benchmark>"),
                                   cl::Required);
cl::opt<uint64_t> BenchmarkSize(cl::Positional, cl::desc("<n>"),
                                cl::init(50000));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  // Initialize.
  MLIRContext ctx;
  ctx.allowUnregisteredDialects();
  ctx.getOrLoadDialect<arith::ArithDialect>();

  runBench(ctx, BenchmarkMode, BenchmarkSize);

  return 0;
}
