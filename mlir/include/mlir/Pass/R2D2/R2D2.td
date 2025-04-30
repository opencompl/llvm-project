#ifndef MLIR_DIALECT_R2D2_IR_R2D2
#define MLIR_DIALECT_R2D2_IR_R2D2

include "mlir/IR/OpBase.td"

def R2D2_Dialect : Dialect 
{
  let name = "r2d2";
  let cppNamespace = "::mlir::r2d2";

  let useDefaultTypePrinterParser = 1;
}

#endif // MLIR_DIALECT_R2D2_IR_R2D2