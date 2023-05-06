//===- MlirOptMain.cpp - MLIR Optimizer Driver ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a utility that runs an optimization pass and prints the result back
// out. It is designed to support unit testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-irdlgen/MlirIrdlGenMain.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/Debug/Counter.h"
#include "mlir/Debug/DebuggerExecutionContextHook.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/Debug/Observers/ActionLogging.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"


using namespace mlir;
using namespace llvm;

void insertDialect(MLIRContext &ctx, OpBuilder &builder, StringRef name) {
  ctx.getOrLoadDialect(name);

  // Create the IDRL dialect operation, and set the insertion point in it.
  auto dialectDef = builder.create<irdl::DialectOp>(
      UnknownLoc::get(&ctx), StringAttr::get(&ctx, name));
  auto &dialectBlock = dialectDef.getBody().emplaceBlock();
  builder.setInsertionPoint(&dialectBlock, dialectBlock.begin());

  for (auto op : ctx.getRegisteredOperations()) {
    StringRef opName = op.getStringRef();
    if (!opName.starts_with(name))
       continue;

    opName = opName.drop_front(name.size() + 1);

    auto opc = builder.create<irdl::OperationOp>(UnknownLoc::get(&ctx), StringAttr::get(&ctx, opName));
    opc.getBody().emplaceBlock();
  }
}

LogicalResult mlir::mlirIrdlGenMain(int argc, char **argv,
                                    MLIRContext &ctx) {

  ctx.getOrLoadDialect<irdl::IRDLDialect>();

  auto unknownLoc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module(ModuleOp::create(unknownLoc));

  // Create the builder, and set its insertion point in the module.
  OpBuilder builder(&ctx);
  auto &moduleBlock = module->getRegion().getBlocks().front();

  for (auto dialect : ctx.getAvailableDialects()) {
    builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());
    insertDialect(ctx, builder, dialect);
  }
 
  module->print(llvm::outs());

  return success();
}
