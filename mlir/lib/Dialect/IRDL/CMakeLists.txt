add_mlir_dialect_library(MLIRIRDL
  IR/IRDL.cpp
  IR/IRDLAttributes.cpp
  IR/IRDLOps.cpp
  IRDLRegistration.cpp
  IRDLVerifiers.cpp

  DEPENDS
  MLIRIRDLIncGen
  MLIRIRDLOpsIncGen
  MLIRIRDLTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
