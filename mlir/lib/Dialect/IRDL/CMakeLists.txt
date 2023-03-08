add_mlir_dialect_library(MLIRIRDL
  IR/IRDL.cpp
  IR/IRDLAttributes.cpp
  IR/IRDLOps.cpp
  AttributeWrapper.cpp
  IRDLLoading.cpp  
  IRDLVerifiers.cpp
  IRDLContext.cpp

  DEPENDS
  MLIRIRDLIncGen
  MLIRIRDLOpsIncGen
  MLIRIRDLTypesIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  )
