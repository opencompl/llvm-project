//===- IsDynamicInterfaces.td - Dynamic objects interfaces -*- tablegen -*-===//
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

#ifndef MLIR_IR_ISDYNAMICINTERFACES
#define MLIR_IR_ISDYNAMICINTERFACES

include "mlir/IR/OpBase.td"

def IsDynamicTypeInterface : TypeInterface<"IsDynamicTypeInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    This interface is implemented by all dynamic types, and should not be
    implemented by any other type.
    The interface is used to check if a type is a dynamic type or not.
  }];

  let methods = [];
}

#endif // MLIR_IR_ISDYNAMICINTERFACES