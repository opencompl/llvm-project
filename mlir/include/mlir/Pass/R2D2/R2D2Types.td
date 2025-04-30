#ifndef MLIR_R2D2_TYPES
#define MLIR_R2D2_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/Pass/R2D2/R2D2.td"
include "mlir/IR/CommonAttrConstraints.td"

class R2D2_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<R2D2_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def R2D2_LocationType : R2D2_Type<"Location", "location"> {
  let summary = "yolo";
}

def R2D2_SnapshotType : R2D2_Type<"Snapshot", "snapshot">  {
  let summary = "yolo";
}

#endif // MLIR_R2D2_TYPES