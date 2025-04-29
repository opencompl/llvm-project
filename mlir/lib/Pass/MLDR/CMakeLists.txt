add_mlir_library(MLIRMLDR
  MLDR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Pass/MLDR

  DEPENDS
  mlir-generic-headers
  MLIRMLDROpsIncGen
  MLIRMLDRTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  )
