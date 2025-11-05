set(LLVM_LINK_COMPONENTS
  TableGen
)

add_tablegen(test-asymptotic MLIR
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT MLIR
  test-asymptotic.cpp
  )

target_link_libraries(test-asymptotic
  PRIVATE
  MLIRIR
  MLIRArithDialect
  MLIRTblgenLib
  MLIRSupport
)

mlir_check_all_link_libraries(test-asymptotic)
