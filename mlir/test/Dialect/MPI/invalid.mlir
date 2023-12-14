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

// -----

func.func @mpi_test(%ref : memref<100xf32>, %rank: i32) -> () {
    // expected-error @+1 {{'mpi.recv' op result #0 must be MPI function call return value, but got 'i32'}}
    %res = mpi.recv(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> i32

    return
}

// -----

func.func @mpi_test(%ref : memref<100xf32>, %rank: i32) -> () {
    // expected-error @+1 {{'mpi.send' op result #0 must be MPI function call return value, but got 'i32'}}
    %res = mpi.send(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> i32

    return
}

// -----

func.func @mpi_test(%retval: !mpi.retval) -> () {
    // expected-error @+2 {{custom op 'mpi.retval_check' expected ::mlir::mpi::MpiErrorClassEnum}}
    // expected-error @+1 {{custom op 'mpi.retval_check' failed to parse MpiErrorClassAttr parameter 'value'}}
    %res = mpi.retval_check %retval = <MPI_ERR_DOES_NOT_EXIST>

    return
}
