add_mlir_library(MLIRTestIRDLToCppDialect
  TestIRDLToCppDialect.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRTestAttrDefIncGen
  MLIRTestEnumDefIncGen
  MLIRTestInterfaceIncGen
  MLIRTestTypeDefIncGen
  MLIRTestOpsIncGen
  MLIRTestOpsSyntaxIncGen
  MLIRTestOpsShardGen

  LINK_LIBS PUBLIC
  MLIRIR
)