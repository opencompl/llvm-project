#!/bin/sh
BUILD_DIR=../build
alias mlir-r2d2-server=${BUILD_DIR}/bin/mlir-r2d2-server

cmake --build ${BUILD_DIR} &&
  echo "running server" && 
  mlir-r2d2-server < payload.json && 
  echo hi