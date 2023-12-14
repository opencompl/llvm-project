// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

func.func @mpi_test(%ref : memref<100xf32>) -> () {
    // CHECK: %0 = mpi.init : !mpi.retval
    %err = mpi.init : !mpi.retval

    // CHECK-NEXT: %retval, %rank = mpi.comm_rank : !mpi.retval, i32
    %retval, %rank = mpi.comm_rank : !mpi.retval, i32

    // CHECK-NEXT: mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32
    mpi.send(%ref, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32
    mpi.recv(%ref, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: %1 = mpi.finalize : !mpi.retval
    %rval = mpi.finalize : !mpi.retval

    // CHECK-NEXT: %2 = mpi.retval_check %retval = <MPI_SUCCESS> : i1
    %res = mpi.retval_check %retval = <MPI_SUCCESS> : i1

    // CHECK-NEXT: %3 = mpi.error_class %0 : !mpi.retval
    %errclass = mpi.error_class %err : !mpi.retval

    // CHECK-NEXT: return
    func.return
}
