set(LIBS
  MLIRR2D2ServerLib
  )

add_mlir_tool(mlir-r2d2-server
  mlir-r2d2-server.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(mlir-r2d2-server PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-r2d2-server)

mlir_check_all_link_libraries(mlir-r2d2-server)
