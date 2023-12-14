//===- MPI.td - Message Passing Interface Ops --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MPI_MLIR_IR_MPIOPS
#define MPI_MLIR_IR_MPIOPS

include "mlir/Dialect/MPI/IR/MPI.td"
include "mlir/Dialect/MPI/IR/MPITypes.td"

class MPI_Op<string mnemonic, list<Trait> traits = []>
    : Op<MPI_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// InitOp
//===----------------------------------------------------------------------===//

def MPI_InitOp : MPI_Op<"init", []> {
  let summary =
      "Initialize the MPI library, equivalent to `MPI_Init(NULL, NULL)`";
  let description = [{
    This operation must preceed most MPI calls (except for very few exceptions,
    please consult with the MPI specification on these).

    Passing &argc, &argv is not supported currently.
  }];

  let results = (outs Optional<MPI_Retval>:$retval);

  let assemblyFormat = "attr-dict (`:` type($retval)^)?";
}

//===----------------------------------------------------------------------===//
// CommRankOp
//===----------------------------------------------------------------------===//

def MPI_CommRankOp : MPI_Op<"comm_rank", [

]> {
  let summary = "Get the current rank, equivalent to "
                "`MPI_Comm_rank(MPI_COMM_WORLD, &rank)`";
  let description = [{
    Communicators other than `MPI_COMM_WORLD` are not supprted for now.
    Inspecting the functions return value (error code) is also not supported.
  }];

  let results = (outs I32 : $result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// SendOp
//===----------------------------------------------------------------------===//

def MPI_SendOp : MPI_Op<"send", [

]> {
  let summary =
      "Equivalent to `MPI_Send(ptr, size, dtype, dest, tag, MPI_COMM_WORLD)`";
  let description = [{
    MPI_Send performs a blocking send of `size` elements of type `dtype` to rank `dest`.
    The `tag` value and communicator enables the library to determine the matching of
    multiple sends and receives between the same ranks.

    Communicators other than `MPI_COMM_WORLD` are not supprted for now.
    Inspecting the functions return value (error code) is also not supported.
  }];

  let arguments = (ins AnyMemRef : $ref, I32 : $tag, I32 : $rank);

  let assemblyFormat = "`(` $ref `,` $tag `,` $rank `)` attr-dict `:` "
                       "type($ref) `,` type($tag) `,` type($rank)";
}

//===----------------------------------------------------------------------===//
// RecvOp
//===----------------------------------------------------------------------===//

def MPI_RecvOp : MPI_Op<"recv", [

]> {
  let summary = "Equivalent to `MPI_Recv(ptr, size, dtype, dest, tag, "
                "MPI_COMM_WORLD, MPI_STATUS_IGNORE)`";
  let description = [{
    MPI_Recv performs a blocking receive of `size` elements of type `dtype` from rank `dest`.
    The `tag` value and communicator enables the library to determine the matching of
    multiple sends and receives between the same ranks.

    Communicators other than `MPI_COMM_WORLD` are not supprted for now.
    The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object is not yet ported to MLIR.
    Inspecting the functions return value (error code) is also not supported.
  }];

  let arguments = (ins AnyMemRef : $ref, I32 : $tag, I32 : $rank);

  let assemblyFormat = "`(` $ref `,` $tag `,` $rank `)` attr-dict `:` "
                       "type($ref) `,` type($tag) `,` type($rank)";
}


//===----------------------------------------------------------------------===//
// FinalizeOp
//===----------------------------------------------------------------------===//

def MPI_FinalizeOp : MPI_Op<"finalize", [

]> {
  let summary = "Finalize the MPI library, equivalent to `MPI_Finalize()`";
  let description = [{
    This function cleans up the MPI state. Afterwards, no MPI methods may be invoked
    (excpet for MPI_Get_version, MPI_Initialized, and MPI_Finalized).
    Notably, MPI_Init cannot be called again in the same program.

    Inspecting the functions return value (error code) is not supported.
  }];

  let assemblyFormat = "attr-dict";
}


//===----------------------------------------------------------------------===//
// FinalizeOp
//===----------------------------------------------------------------------===//

def MPI_RetvalCheck : MPI_Op<"retval_check", [

]> {
  let summary = "Check an MPI return value against an error class";
  let description = [{
    
  }];

  let arguments = (
    ins MPI_Retval:$val,
    MpiErrorClass:$errclass
  );

  let assemblyFormat = "$val `=` $errclass attr-dict";
}

#endif // MPI_MLIR_IR_MPIOPS
