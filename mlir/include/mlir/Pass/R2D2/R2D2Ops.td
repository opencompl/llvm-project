#ifndef R2D2_OPS
#define R2D2_OPS

include "mlir/Pass/R2D2/R2D2.td"
include "mlir/Pass/R2D2/R2D2Types.td"
include "mlir/IR/CommonAttrConstraints.td"

def R2D2_LocationOp : Op<R2D2_Dialect, "loc", [NoRegionArguments]> {
  let arguments = (ins  R2D2_SnapshotType: $snapshot, 
                        I32Attr: $line, 
                        I32Attr: $column, 
                        Variadic<R2D2_LocationType>: $args);
  let results = (outs R2D2_LocationType: $result);
  
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

def R2D2_PassOp : Op<R2D2_Dialect, "pass", [NoRegionArguments]> {
  let arguments = (ins StrAttr: $snapshot,
                       R2D2_SnapshotType: $prev);
  let results = (outs R2D2_SnapshotType: $result);
  
  let assemblyFormat = [{
    $snapshot `(` $prev `)` attr-dict
  }];

  let builders = [
    OpBuilder<(ins "StringAttr": $snapshot, "::mlir::Value": $prev)>,
    OpBuilder<(ins "StringRef": $snapshot, "::mlir::Value": $prev)>
  ];
}

def R2D2_TraceOp : Op<R2D2_Dialect, "trace", [NoTerminator, IsolatedFromAbove]> {
  let arguments = (ins StrAttr: $snapshot);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

#endif // R2D2_OPS