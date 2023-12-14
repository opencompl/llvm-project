//===- MPITypes.td - Message Passing Interface types -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Message Passing Interface dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MPI_IR_MPITYPES
#define MLIR_DIALECT_MPI_IR_MPITYPES

include "mlir/IR/AttrTypeBase.td"
include "MPI.td"

//===----------------------------------------------------------------------===//
// NPI Types
//===----------------------------------------------------------------------===//

class MPI_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MPI_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// pdl::AttributeType
//===----------------------------------------------------------------------===//

def MPI_Retval : MPI_Type<"Retval", "retval"> {
  let summary = "The return value of an MPI call.";
  let description = [{
    This type represents a value returned from an MPI call. This value can be
    MPI_SUCCESS, MPI_ERR_IN_STATUS, or any error code.
  }];
}


#endif // MLIR_DIALECT_MPI_IR_MPITYPES