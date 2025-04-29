add_mlir_dialect(MLDR mldr -dialect=mldr)

set(LLVM_TARGET_DEFINITIONS MLDROps.td)
mlir_tablegen(MLDROps.h.inc -gen-op-decls -dialect=mldr)
mlir_tablegen(MLDROps.cpp.inc -gen-op-defs -dialect=mldr)
add_public_tablegen_target(MLIRMLDROpsIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLOpsIncGen)


set(LLVM_TARGET_DEFINITIONS MLDRTypes.td)
mlir_tablegen(MLDRTypesGen.h.inc -gen-typedef-decls)
mlir_tablegen(MLDRTypesGen.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRMLDRTypesIncGen)
add_dependencies(mlir-generic-headers MLIRMLDRTypesIncGen)
