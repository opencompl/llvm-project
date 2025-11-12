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

static Operation *rewriteAddConstantFolding(Operation *op) {
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

struct AddConstantFoldingPattern : public OpRewritePattern<arith::AddIOp> {
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

static Operation *rewriteAddZero(Operation *op) {
  auto addOp = dyn_cast<arith::AddIOp>(op);
  if (!addOp)
    return nullptr;
  auto lhs = addOp.getLhs();
  auto rhs = addOp.getRhs();

  auto cst0 = lhs.getDefiningOp<arith::ConstantIntOp>();
  if (!cst0)
    return nullptr;

  if (cst0.value() != 0)
    return nullptr;

  addOp.replaceAllUsesWith(rhs);
  addOp.erase();

  if (lhs.getUses().empty())
    lhs.getDefiningOp()->erase();

  return nullptr;
}

struct AddZeroPattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::AddIOp addOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    auto cst = lhs.getDefiningOp<arith::ConstantIntOp>();
    if (!cst)
      return failure();

    if (cst.value() != 0)
      return failure();

    rewriter.replaceOp(addOp, rhs);
    if (lhs.getUses().empty())
      rewriter.eraseOp(lhs.getDefiningOp());
    return success();
  }
};

static Operation *rewriteMul2Reduce(Operation *op) {
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

struct Mul2ReducePattern : public OpRewritePattern<arith::MulIOp> {
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

template <typename F>
static void rewriteModule(ModuleOp module, F f) {
  Operation *op = &module.getBody()->getOperations().front();
  while (op) {
    Operation *next = op->getNextNode();
    auto *newOp = f(op);
    if (newOp) {
      op = newOp;
      continue;
    }
    op = next;
  }
}

template <typename Rewrite>
static void rewriteModuleWithPatternRewriter(ModuleOp module) {
  auto *ctx = module->getContext();

  mlir::GreedyRewriteConfig config;
  config.cseConstants = false;
  config.fold = false;

  mlir::RewritePatternSet patterns(ctx);
  patterns.insert<Rewrite>(ctx);
  (void)mlir::applyPatternsGreedily(module, std::move(patterns), config);
}

static OwningOpRef<ModuleOp> createModule(MLIRContext &ctx, uint64_t size,
                                          uint32_t root, uint32_t increment) {
  OpBuilder builder(&ctx);
  OwningOpRef<ModuleOp> module =
      builder.create<ModuleOp>(UnknownLoc::get(&ctx));
  builder = builder.atBlockBegin(module->getBody());

  Operation *rootOp = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx),
      IntegerAttr::get(IntegerType::get(&ctx, 32), root));
  for (uint64_t i = 0; i < size; ++i) {
    auto cst0 = builder.create<arith::ConstantOp>(
        UnknownLoc::get(&ctx),
        IntegerAttr::get(IntegerType::get(&ctx, 32), increment));
    rootOp = builder.create<arith::AddIOp>(
        UnknownLoc::get(&ctx), cst0->getResult(0), rootOp->getResult(0));
  }
  OperationState state(UnknownLoc::get(&ctx), "test.test");
  state.addOperands(rootOp->getResult(0));
  builder.create(state);

  return module;
}

static OwningOpRef<ModuleOp> createMul2Module(MLIRContext &ctx, uint64_t size,
                                              uint32_t root) {
  OpBuilder builder(&ctx);
  OwningOpRef<ModuleOp> module =
      builder.create<ModuleOp>(UnknownLoc::get(&ctx));
  builder = builder.atBlockBegin(module->getBody());

  Operation *rootOp = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx),
      IntegerAttr::get(IntegerType::get(&ctx, 32), root));

  for (uint64_t i = 0; i < size; ++i) {
    auto cst1 = builder.create<arith::ConstantOp>(
        UnknownLoc::get(&ctx), IntegerAttr::get(IntegerType::get(&ctx, 32), 2));

    rootOp = builder.create<arith::MulIOp>(
        UnknownLoc::get(&ctx), rootOp->getResult(0), cst1->getResult(0));
  }

  OperationState state(UnknownLoc::get(&ctx), "test.test");
  state.addOperands(rootOp->getResult(0));
  builder.create(state);

  return module;
}

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

  if (BenchmarkMode == "constant-folding") {
    OwningOpRef<ModuleOp> module = time(
        "create", [&]() { return createModule(ctx, BenchmarkSize, 42, 1); });
    time("rewrite",
         [&]() { rewriteModule(*module, rewriteAddConstantFolding); });
    module->dump();
  } else if (BenchmarkMode == "constant-folding-rewriter") {
    OwningOpRef<ModuleOp> module = time(
        "create", [&]() { return createModule(ctx, BenchmarkSize, 42, 1); });
    time("rewrite", [&]() {
      rewriteModuleWithPatternRewriter<AddConstantFoldingPattern>(*module);
    });
    module->dump();

  } else if (BenchmarkMode == "add-zero") {
    OwningOpRef<ModuleOp> module = time(
        "create", [&]() { return createModule(ctx, BenchmarkSize, 42, 0); });
    time("rewrite", [&]() { rewriteModule(*module, rewriteAddZero); });
    module->dump();
  } else if (BenchmarkMode == "add-zero-rewriter") {
    OwningOpRef<ModuleOp> module = time(
        "create", [&]() { return createModule(ctx, BenchmarkSize, 42, 0); });
    time("rewrite",
         [&]() { rewriteModuleWithPatternRewriter<AddZeroPattern>(*module); });
    module->dump();

  } else if (BenchmarkMode == "mul2-reduce") {
    OwningOpRef<ModuleOp> module = time(
        "create", [&]() { return createMul2Module(ctx, BenchmarkSize, 42); });
    time("rewrite", [&]() { rewriteModule(*module, rewriteMul2Reduce); });
  } else if (BenchmarkMode == "mul2-reduce-rewriter") {
    OwningOpRef<ModuleOp> module = time(
        "create", [&]() { return createMul2Module(ctx, BenchmarkSize, 42); });
    time("rewrite", [&]() {
      rewriteModuleWithPatternRewriter<Mul2ReducePattern>(*module);
    });
  } else {
    llvm::errs() << "Unrecognised benchmark name\n";
    return EXIT_FAILURE;
  }

  return 0;
}
