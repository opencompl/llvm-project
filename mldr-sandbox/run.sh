#!/bin/sh
cmake --build ../build &&
../build/bin/mlir-opt -debug --pass-pipeline="builtin.module(test-constant-mul,snapshot-op-locations)" test.mlir