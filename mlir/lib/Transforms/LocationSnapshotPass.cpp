//===- LocationSnapshot.cpp - Location Snapshot Utilities -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/LocationSnapshotPass.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_LOCATIONSNAPSHOT
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct LocationSnapshotPass
    : public impl::LocationSnapshotBase<LocationSnapshotPass> {
  using impl::LocationSnapshotBase<LocationSnapshotPass>::LocationSnapshotBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    if (failed(generateLocationsFromIR(fileName, tag, op, getFlags())))
      return signalPassFailure();
  }

private:
  /// build the flags from the command line arguments to the pass
  OpPrintingFlags getFlags() {
    OpPrintingFlags flags;
    flags.enableDebugInfo(enableDebugInfo, printPrettyDebugInfo);
    flags.printGenericOpForm(printGenericOpForm);
    if (useLocalScope)
      flags.useLocalScope();
    return flags;
  }
};
} // namespace
