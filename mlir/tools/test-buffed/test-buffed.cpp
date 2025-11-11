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

template <typename F>
static void rewriteModule(ModuleOp module, F f) {
  Operation *op = &module.getBody()->getOperations().front();
  while (op) {
    Operation* next = op->getNextNode();
    auto *newOp = f(op);
    if (newOp) {
      op = newOp;
      continue;
    }
    op = next;
  }
}

static OwningOpRef<ModuleOp> createModule(MLIRContext &ctx, uint64_t size, uint32_t root, uint32_t increment) {
  OpBuilder builder(&ctx);
  OwningOpRef<ModuleOp> module =
      builder.create<ModuleOp>(UnknownLoc::get(&ctx));
  builder = builder.atBlockBegin(module->getBody());

  Operation *rootOp = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx), IntegerAttr::get(IntegerType::get(&ctx, 32), root));
  for (uint64_t i = 0; i < size; ++i) {
    auto cst0 = builder.create<arith::ConstantOp>(
        UnknownLoc::get(&ctx), IntegerAttr::get(IntegerType::get(&ctx, 32), increment));
    rootOp = builder.create<arith::AddIOp>(
        UnknownLoc::get(&ctx), cst0->getResult(0), rootOp->getResult(0));
  }
  return module;
}

template<typename F>
static auto time(std::string_view name, F f) {
  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();

  const auto print_time = [&]() {
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    llvm::errs() << name << " time (s): "
                 << elapsed / std::chrono::seconds(1)
                 << "\n";
  };

  if constexpr(std::is_void_v<decltype(f())>) {
    f();
    print_time();
    return;
  } else {
    auto ret = f();
    print_time();
    return ret;
  }
}

cl::opt<std::string> BenchmarkMode(cl::Positional, cl::desc("<benchmark>"), cl::Required);
cl::opt<uint64_t> BenchmarkSize(cl::Positional, cl::desc("<n>"), cl::init(50000));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  // Initialize.
  MLIRContext ctx;
  ctx.getOrLoadDialect<arith::ArithDialect>();

  if (BenchmarkMode == "constant-folding") {
    OwningOpRef<ModuleOp> module = time("create", [&]() {
        return createModule(ctx, BenchmarkSize, 42, 1);
    });
    time("rewrite", [&]() { rewriteModule(*module, rewriteAddConstantFolding); });
    module->dump();
  }
  else if (BenchmarkMode == "add-zero") {
    OwningOpRef<ModuleOp> module = time("create", [&]() {
        return createModule(ctx, BenchmarkSize, 42, 0);
    });
    time("rewrite", [&]() { rewriteModule(*module, rewriteAddZero); });
    module->dump();
  } else {
    llvm::errs() << "Unrecognised benchmark name\n";
    return EXIT_FAILURE;
  }

  return 0;
}
