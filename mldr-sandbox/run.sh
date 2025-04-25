#!/bin/sh
BUILD_DIR=../build
alias mlir-opt=${BUILD_DIR}/bin/mlir-opt

cmake --build ${BUILD_DIR} &&
  mlir-opt -mlir-print-debuginfo -mlir-snapshot-after-all --pass-pipeline="builtin.module\
    ( test-constant-mul \
    )" test.mlir > snapshot-out.mlir