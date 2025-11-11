set(LLVM_LINK_COMPONENTS
  TableGen
)

add_tablegen(test-buffed MLIR
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT MLIR
  test-buffed.cpp
  )

target_link_libraries(test-buffed
  PRIVATE
  MLIRIR
  MLIRArithDialect
  MLIRTblgenLib
  MLIRSupport
  MLIRTransforms
)

mlir_check_all_link_libraries(test-buffed)
