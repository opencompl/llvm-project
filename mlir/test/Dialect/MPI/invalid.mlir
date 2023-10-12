// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// expected-error @+1 {{op result #0 must be 32-bit signless integer, but got 'i64'}}
%rank = mpi.comm_rank : i64

// -----

func.func @mpi_test(%ref : !llvm.ptr, %rank: i32) -> () {
    // expected-error @+1 {{invalid kind of type specified}}
    mpi.send(%ref, %rank, %rank) : !llvm.ptr, i32, i32

    return
}

// -----

func.func @mpi_test(%ref : !llvm.ptr, %rank: i32) -> () {
    // expected-error @+1 {{invalid kind of type specified}}
    mpi.recv(%ref, %rank, %rank) : !llvm.ptr, i32, i32

    return
}
