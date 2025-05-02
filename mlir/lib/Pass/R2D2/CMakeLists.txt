add_mlir_library(MLIRR2D2
    R2D2.cpp
    R2D2Support.cpp

    ADDITIONAL_HEADER_DIRS 
    ${MLIR_MAIN_INCLUDE_DIR}
    /mlir/Pass/R2D2

    DEPENDS
    mlir-generic-headers
    MLIRR2D2OpsIncGen
    MLIRR2D2TypesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)
