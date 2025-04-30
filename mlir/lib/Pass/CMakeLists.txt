add_subdirectory(R2D2)

add_mlir_library(MLIRPass
  IRPrinting.cpp
  Pass.cpp
  PassCrashRecovery.cpp
  PassManagerOptions.cpp
  PassRegistry.cpp
  PassSnapshot.cpp
  PassStatistics.cpp
  PassTiming.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Pass

  DEPENDS
  mlir-generic-headers

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRSupport
  MLIRR2D2
  )
