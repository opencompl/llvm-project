#ifndef MLIR_PASS_R2D2_R2D2_H
#define MLIR_PASS

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Pass/R2D2/R2D2.h.inc"
#include "mlir/Pass/R2D2/R2D2Dialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Pass/R2D2/R2D2TypesGen.h.inc"
#define GET_OP_CLASSES
#include "mlir/Pass/R2D2/R2D2Ops.h.inc"

#endif