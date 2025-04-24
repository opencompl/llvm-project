#!/bin/sh
cmake --build ../build &&
../build/bin/mlir-opt -mlir-print-debuginfo --pass-pipeline="builtin.module(test-constant-mul, snapshot-op-locations{filename=snapshot.mlir})"  test.mlir