set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  )

# Exclude from libMLIR.so because this has static options intended for
# opt-like tools only.
add_mlir_library(MLIRMlirIrdlOptMain
  mlir-irdl-opt.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ${LIBS}
  )

add_mlir_tool(mlir-irdl-opt
  mlir-irdl-opt.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(mlir-irdl-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-irdl-opt)

mlir_check_all_link_libraries(mlir-irdl-opt)
