//===- IsDynamicInterfaces.h - Dynamic objects interfaces --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Define an interface that is only implemented on dynamic types.
// The interface is used to check if a type is a DynamicType or not.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ISDYNAMICINTERFACES_H
#define MLIR_IR_ISDYNAMICINTERFACES_H

#include "mlir/IR/OpDefinition.h"

//===----------------------------------------------------------------------===//
// Type interfaces
//===----------------------------------------------------------------------===//

#include "mlir/IR/IsDynamicTypeInterfaces.h.inc"

#endif // MLIR_IR_ISDYNAMICINTERFACES_H
