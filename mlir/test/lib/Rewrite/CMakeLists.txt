if (MLIR_ENABLE_PDL_IN_PATTERNMATCH)
  # Exclude tests from libMLIR.so
  add_mlir_library(MLIRTestRewrite
    TestPDLByteCode.cpp
    TestRewriteRecorder.cpp

    EXCLUDE_FROM_LIBMLIR

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rewrite
    )
  mlir_target_link_libraries(MLIRTestRewrite PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransformUtils
    MLIRArithDialect
    )
endif()
