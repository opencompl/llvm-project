//===- MPIDialect.cpp - MPI dialect implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::mpi;

//===----------------------------------------------------------------------===//
/// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MPI/IR/MPIOpsDialect.cpp.inc"

void mpi::MPIDialect::initialize() {

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MPI/IR/MPIOps.cpp.inc"
      >();
}
