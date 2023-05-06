add_mlir_library(MLIRIrdlGenLib
  MlirIrdlGenMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-irdlgen

  LINK_LIBS PUBLIC
  MLIRBytecodeWriter
  MLIRDebug
  MLIRObservers
  MLIRPass
  MLIRParser
  MLIRPluginsLib
  MLIRSupport
  MLIRIRDL
  )
