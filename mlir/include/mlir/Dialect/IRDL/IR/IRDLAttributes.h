//===- IRDLAttributes.h - Attributes definition for IRDL --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes used in the IRDL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDLATTRIBUTES_H_
#define MLIR_DIALECT_IRDL_IR_IRDLATTRIBUTES_H_

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"

// Forward declarations
namespace mlir {
namespace irdl {
class TypeOrParamTypeAttr;
}
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.h.inc"

#endif // MLIR_DIALECT_IRDL_IR_IRDLATTRIBUTES_H_
