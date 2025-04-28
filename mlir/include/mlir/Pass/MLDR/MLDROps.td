#ifndef MLDR_OPS
#define MLDR_OPS

include "mlir/Pass/MLDR/MLDR.td"
include "mlir/Pass/MLDR/MLDRTypes.td"
include "mlir/IR/CommonAttrConstraints.td"

def MLDR_LocationOp : Op<MLDR_Dialect, "loc", []> {
  let arguments = (ins  MLDR_PassType: $pass, 
                        I32Attr: $line, 
                        I32Attr: $col, 
                        Variadic<MLDR_LocationType>: $args);
  let results = (outs MLDR_LocationType: $result);
  
  let assemblyFormat = [{
      $pass `:` $line `:` $col `(` $args `)` attr-dict
    }];
}

def MLDR_PassOp : Op<MLDR_Dialect, "pass", []> {
  let arguments = (ins StrAttr: $file,
                       MLDR_PassType: $prev);
  let results = (outs MLDR_PassType: $result);
  
  let assemblyFormat = [{
    $file `(` $prev `)` attr-dict
  }];
}

// def MLDR_TraceOp : Op<MLDR_Dialect, "trace", []> {
//   let regions = (region SizedRegion<1>:$body);
// }

#endif // MLDR_OPS