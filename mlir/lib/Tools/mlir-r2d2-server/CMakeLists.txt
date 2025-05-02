add_mlir_library(MLIRR2D2Server 
  R2D2Server.cpp
  Protocol.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-r2d2-server

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRPass
)
