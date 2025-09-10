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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <chrono>

using namespace llvm;
using namespace mlir;

Operation *matchAndRewrite(Operation *op) {
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

void rewriteModule(ModuleOp module) {
  Operation *op = &module.getBody()->getOperations().front();
  while (op) {
    auto *newOp = matchAndRewrite(op);
    if (newOp) {
      op = newOp;
      continue;
    }
    op = op->getNextNode();
  }
}

OwningOpRef<ModuleOp> createModule(MLIRContext &ctx, int32_t size) {
  OpBuilder builder(&ctx);
  OwningOpRef<ModuleOp> module =
      builder.create<ModuleOp>(UnknownLoc::get(&ctx));
  builder = builder.atBlockBegin(module->getBody());

  Operation *rootOp = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx), IntegerAttr::get(IntegerType::get(&ctx, 32), 1));
  for (int i = 0; i < size / 2; ++i) {
    auto cst0 = builder.create<arith::ConstantOp>(
        UnknownLoc::get(&ctx), IntegerAttr::get(IntegerType::get(&ctx, 32), 1));
    rootOp = builder.create<arith::AddIOp>(
        UnknownLoc::get(&ctx), rootOp->getResult(0), cst0->getResult(0));
  }
  return module;
}

int main(int argc, char **argv) {
  // Initialize.
  MLIRContext ctx;
  ctx.getOrLoadDialect<arith::ArithDialect>();

  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();
  int32_t size = 10'000'000;

  OwningOpRef<ModuleOp> module = createModule(ctx, size);
  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  llvm::errs() << "Time taken to create the program: "
               << elapsed / std::chrono::milliseconds(1) << " ms"
               << "\n";
  llvm::errs() << "Ns per operation: "
               << (elapsed / std::chrono::nanoseconds(1)) / size << " ns \n";

  start = std::chrono::high_resolution_clock::now();
  rewriteModule(*module);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  llvm::errs() << "Time taken to rewrite the program: "
               << elapsed / std::chrono::milliseconds(1) << " ms"
               << "\n";
  llvm::errs() << "Ns per operation: "
               << (elapsed / std::chrono::nanoseconds(1)) / size << "\n";
  llvm::errs() << "Final program:" << "\n";

  llvm::errs() << "Sanity check\n";
  module->dump();

  return 0;
}
