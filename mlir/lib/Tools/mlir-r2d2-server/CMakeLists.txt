add_mlir_library(MLIRR2D2ServerLib
  MlirR2D2ServerMain.cpp
  Protocol.cpp
  R2D2Server.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-r2d2-server

  LINK_COMPONENTS 
  Support

  LINK_LIBS
  MLIRParser
  MLIRPass
  MLIRLspServerSupportLib
)
