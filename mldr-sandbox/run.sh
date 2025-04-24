#!/bin/sh
BUILD_DIR=../build
alias mlir-opt=${BUILD_DIR}/bin/mlir-opt

cmake --build ${BUILD_DIR} &&
  mlir-opt -mlir-print-debuginfo --pass-pipeline="builtin.module\
    ( test-constant-mul \
    , snapshot-op-locations{print-debuginfo=1 pretty-debuginfo=1 filename=snapshot-0.mlir} \
    )" test.mlir > snapshot-out.mlir