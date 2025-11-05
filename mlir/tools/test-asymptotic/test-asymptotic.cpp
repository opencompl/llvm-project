//===- OpDefinitionsGen.cpp - IRDL op definitions generator ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// TODO: variable num args
OwningOpRef<ModuleOp> testCreateFixedArgs(MLIRContext& ctx, const size_t size) {
  OpBuilder builder(&ctx);
  OwningOpRef<ModuleOp> module = builder.create<ModuleOp>(UnknownLoc::get(&ctx));
  builder = builder.atBlockBegin(module->getBody());

  Operation* acc = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx), IntegerAttr::get(IntegerType::get(&ctx, 32), 1));

  for (size_t i = 0; i < size; i++) {
    acc = builder.create<arith::AddIOp>(
        UnknownLoc::get(&ctx), acc->getResult(0), acc->getResult(0));
  }

  return module;
}

// We build up a module by inserting constants repeatedly into the program
// alternating outwards from an innermost op
OwningOpRef<ModuleOp> testInsert(MLIRContext& ctx, const size_t size) {
  OpBuilder builder(&ctx);
  OwningOpRef<ModuleOp> module = builder.create<ModuleOp>(UnknownLoc::get(&ctx));
  builder = builder.atBlockBegin(module->getBody());

  Operation* low = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx), IntegerAttr::get(IntegerType::get(&ctx, 32), 1));
  Operation* high = low;

  for (size_t i = 0; i < size / 2; i++) {
    builder.setInsertionPoint(low);
    low = builder.create<arith::ConstantOp>(
        UnknownLoc::get(&ctx), IntegerAttr::get(IntegerType::get(&ctx, 32), i));

    builder.setInsertionPointAfter(high);
    high = builder.create<arith::ConstantOp>(
        UnknownLoc::get(&ctx), IntegerAttr::get(IntegerType::get(&ctx, 32), i));
  }

  return module;
}


int main(int argc, char **argv) {
  // Initialize.
  MLIRContext ctx;
  ctx.getOrLoadDialect<arith::ArithDialect>();

  // auto mod = testCreateFixedArgs(ctx, atoi(argv[1]));
  auto mod = testInsert(ctx, atoi(argv[1]));

  // std::chrono::high_resolution_clock::time_point start =
  //     std::chrono::high_resolution_clock::now();
  // // int32_t size = 50'000;
  // int32_t size = atoi(argv[1]);

  // OwningOpRef<ModuleOp> module = createModule(ctx, size);
  // std::chrono::high_resolution_clock::time_point end =
  //     std::chrono::high_resolution_clock::now();

  // std::chrono::duration<double> elapsed = end - start;
  // llvm::errs() << "Time taken to create the program: "
  //              << elapsed / std::chrono::milliseconds(1) << " ms"
  //              << "\n";
  // llvm::errs() << "Ns per operation: "
  //              << (elapsed / std::chrono::nanoseconds(1)) / size << " ns \n";

  // start = std::chrono::high_resolution_clock::now();
  // rewriteModule(*module);
  // end = std::chrono::high_resolution_clock::now();
  // elapsed = end - start;
  // llvm::errs() << "Time taken to rewrite the program: "
  //              << elapsed / std::chrono::milliseconds(1) << " ms"
  //              << "\n";
  // llvm::errs() << "Ns per operation: "
  //              << (elapsed / std::chrono::nanoseconds(1)) / size << "\n";
  // llvm::errs() << "Final program:" << "\n";

  // llvm::errs() << "Sanity check\n";
  // module->dump();

  return 0;
}
