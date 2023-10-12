set(LLVM_TARGET_DEFINITIONS MPIOps.td)
add_mlir_dialect(MPIOps mpi)
add_mlir_doc(MPIOps MPIOps Dialects/ -gen-dialect-doc)
