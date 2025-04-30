add_mlir_dialect(R2D2 r2d2 -dialect=r2d2)

set(LLVM_TARGET_DEFINITIONS R2D2Ops.td)
mlir_tablegen(R2D2Ops.h.inc -gen-op-decls -dialect=r2d2)
mlir_tablegen(R2D2Ops.cpp.inc -gen-op-defs -dialect=r2d2)
add_public_tablegen_target(MLIRR2D2OpsIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLOpsIncGen)

set(LLVM_TARGET_DEFINITIONS R2D2Types.td)
mlir_tablegen(R2D2TypesGen.h.inc -gen-typedef-decls)
mlir_tablegen(R2D2TypesGen.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRR2D2TypesIncGen)
add_dependencies(mlir-generic-headers MLIRR2D2TypesIncGen)
