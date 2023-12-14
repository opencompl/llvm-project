// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

func.func @mpi_test(%ref : memref<100xf32>) -> () {
    // CHECK: %0 = mpi.init : !mpi.retval
    %err = mpi.init : !mpi.retval

    // CHECK-NEXT: mpi.comm_rank : i32
    %rank = mpi.comm_rank : i32

    // CHECK-NEXT: mpi.send(%arg0, %1, %1) : memref<100xf32>, i32, i32
    mpi.send(%ref, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: mpi.recv(%arg0, %1, %1) : memref<100xf32>, i32, i32
    mpi.recv(%ref, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: mpi.finalize
    mpi.finalize

    func.return
}
