#ifndef MLDR_OPS
#define MLDR_OPS

include "mlir/Pass/MLDR/MLDR.td"
include "mlir/Pass/MLDR/MLDRTypes.td"
include "mlir/IR/CommonAttrConstraints.td"

def MLDR_LocationOp : Op<MLDR_Dialect, "loc", [NoRegionArguments]> {
  let arguments = (ins  MLDR_SnapshotType: $snapshot, 
                        I32Attr: $line, 
                        I32Attr: $column, 
                        Variadic<MLDR_LocationType>: $args);
  let results = (outs MLDR_LocationType: $result);
  
  let assemblyFormat = [{
      `[` $snapshot `:` $line `:` $column `]` `(` $args `)` attr-dict
    }];

  let builders = [
    OpBuilder<(ins "::mlir::Value": $snapshot, "unsigned": $line, "unsigned": $column, CArg<"::mlir::ValueRange", "{}">: $args)>
  ];

  let extraClassDeclaration = [{
    /// Get the relative filepath as a string
    StringAttr getSnapshotFile();

    /// Get the contained location as a FileLineColLoc
    FileLineColLoc getContainedLocation();
  }];
}

def MLDR_PassOp : Op<MLDR_Dialect, "pass", [NoRegionArguments]> {
  let arguments = (ins StrAttr: $snapshot,
                       MLDR_SnapshotType: $prev);
  let results = (outs MLDR_SnapshotType: $result);
  
  let assemblyFormat = [{
    $snapshot `(` $prev `)` attr-dict
  }];

  let builders = [
    OpBuilder<(ins "StringAttr": $snapshot, "::mlir::Value": $prev)>,
    OpBuilder<(ins "StringRef": $snapshot, "::mlir::Value": $prev)>
  ];
}

def MLDR_TraceOp : Op<MLDR_Dialect, "trace", [NoTerminator, IsolatedFromAbove]> {
  let arguments = (ins StrAttr: $snapshot);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

#endif // MLDR_OPS