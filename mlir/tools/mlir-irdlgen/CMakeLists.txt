get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    MLIRTestDialect
    )
endif()

set(LIBS
  ${conversion_libs}
  ${dialect_libs}
  ${test_libs}
  MLIRDialect
  MLIRIR
  MLIRPass
  MLIRIrdlGenLib
  )

add_mlir_tool(mlir-irdlgen
  mlir-irdlgen.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(mlir-irdlgen PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-irdlgen )

mlir_check_all_link_libraries(mlir-irdlgen)
