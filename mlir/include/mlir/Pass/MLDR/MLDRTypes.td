#ifndef MLDR_TYPES
#define MLDR_TYPES

include "mlir/Pass/MLDR/MLDR.td"
include "mlir/IR/CommonAttrConstraints.td"

class MLDR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MLDR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MLDR_LocationType : MLDR_Type<"Location", "location"> {
}

def MLDR_PassType : MLDR_Type<"Pass", "pass">  {
}

#endif // MLDR_TYPES