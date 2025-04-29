#ifndef MLIR_PASS_MLDR_MLDR_H
#define MLIR_PASS

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Pass/MLDR/MLDR.h.inc"
#include "mlir/Pass/MLDR/MLDRDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Pass/MLDR/MLDRTypesGen.h.inc"
#define GET_OP_CLASSES
#include "mlir/Pass/MLDR/MLDROps.h.inc"

#endif