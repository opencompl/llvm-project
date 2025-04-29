#ifndef MLIR_MLDR_TYPES
#define MLIR_MLDR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/Pass/MLDR/MLDR.td"
include "mlir/IR/CommonAttrConstraints.td"

class MLDR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MLDR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MLDR_LocationType : MLDR_Type<"Location", "location"> {
  let summary = "yolo";
}

def MLDR_SnapshotType : MLDR_Type<"Snapshot", "snapshot">  {
  let summary = "yolo";
}

#endif // MLIR_MLDR_TYPES