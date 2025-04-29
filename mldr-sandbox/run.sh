#!/bin/sh
BUILD_DIR=../build
alias mlir-opt=${BUILD_DIR}/bin/mlir-opt

cmake --build ${BUILD_DIR} &&
  mlir-opt  -mlir-snapshot-after-all -mlir-print-debuginfo --pass-pipeline="builtin.module\
    ( test-constant-mul \
    , convert-arith-to-llvm
    )" test.mlir > snapshot-out.mlir
  
# -mlir-print-debuginfo