#ifndef MLIR_DIALECT_MLDR_IR_MLDR
#define MLIR_DIALECT_MLDR_IR_MLDR

include "mlir/IR/OpBase.td"

def MLDR_Dialect : Dialect 
{
  let name = "mldr";
  let cppNamespace = "::mlir::mldr";
  let useDefaultAttributePrinterParser = 1;
}

#endif // MLIR_DIALECT_MLDR_IR_MLDR