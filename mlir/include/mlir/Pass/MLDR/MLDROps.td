#ifndef MLDR_OPS
#define MLDR_OPS

include "mlir/Pass/MLDR/MLDR.td"
include "mlir/Pass/MLDR/MLDRTypes.td"
include "mlir/IR/CommonAttrConstraints.td"

def MLDR_LocationOp : Op<MLDR_Dialect, "loc", []> {
  let arguments = (ins  MLDR_SnapshotType: $pass, 
                        I32Attr: $line, 
                        I32Attr: $col, 
                        Variadic<MLDR_LocationType>: $args);
  let results = (outs MLDR_LocationType: $result);
  
  let assemblyFormat = [{
      `[` $pass `:` $line `:` $col `]` `(` $args `)` attr-dict
    }];
}

def MLDR_PassOp : Op<MLDR_Dialect, "pass", []> {
  let arguments = (ins StrAttr: $snapshot,
                       MLDR_SnapshotType: $prev);
  let results = (outs MLDR_SnapshotType: $result);
  
  let assemblyFormat = [{
    $snapshot `(` $prev `)` attr-dict
  }];
}

def MLDR_TraceOp : Op<MLDR_Dialect, "trace", [NoTerminator, IsolatedFromAbove]> {
  let arguments = (ins StrAttr: $snapshot);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

#endif // MLDR_OPS