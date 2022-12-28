//===- IRDLInterfaces.h - IRDL interfaces definition ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interfaces for the IR Definition Language dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDLINTERFACES_H_
#define MLIR_DIALECT_IRDL_IR_IRDLINTERFACES_H_

#include "mlir/Dialect/IRDL/IRDLContext.h"
#include "mlir/Dialect/IRDL/IRDLConstraint.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

//===----------------------------------------------------------------------===//
// IRDL Dialect Interfaces
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/IRDL/IR/IRDLInterfaces.h.inc"

#endif //  MLIR_DIALECT_IRDL_IR_IRDLINTERFACES_H_
