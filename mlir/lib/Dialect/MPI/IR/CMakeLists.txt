add_mlir_dialect_library(MLIRMPIDialect
  MPIOps.cpp
  MPIDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MPI

  DEPENDS
  MLIRMPIOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  )
