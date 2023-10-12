//===- MPIBase.td - Base defs for mpi dialect --*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MPI_BASE
#define MPI_BASE

include "mlir/IR/OpBase.td"

def MPI_Dialect : Dialect {
  let name = "mpi";
  let cppNamespace = "::mlir::mpi";
  let description = [{
    This dialect models the Message Passing Interface (MPI), version 4.0. It is
    meant to serve as an interfacing dialect that is targeted by higher-level dialects.
    The MPI dialect itself can be lowered to multiple MPI implementations and hide
    differences in ABI. The dialect models the functions of the MPI specification as
    close to 1:1 as possible while preserving SSA value semantics where it makes sense,
    and uses `memref` types instead of bare pointers.

    This dialect is under active development, and while stability is an
    eventual goal, it is not guaranteed at this juncture. Given the early state,
    it is recommended to inquire further prior to using this dialect.

    For an in-depth documentation of the MPI library interface, please refer to official documentation
    such as the [OpenMPI online documentation](https://www.open-mpi.org/doc/current/).
  }];

  let usePropertiesForAttributes = 1;
}

#endif // MPI_BASE
