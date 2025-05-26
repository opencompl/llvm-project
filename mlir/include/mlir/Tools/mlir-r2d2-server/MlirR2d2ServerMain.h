#ifndef MLIR_TOOLS_MLIR_R2D2_SERVER_MLIRR2D2SERVERMAIN_H
#define MLIR_TOOLS_MLIR_R2D2_SERVER_MLIRR2D2SERVERMAIN_H

namespace llvm {
struct LogicalResult;
} // namespace llvm

namespace mlir {
/// Implementation for tools like `mlir-pdll-lsp-server`.
llvm::LogicalResult MlirR2D2ServerMain(int argc, char **argv);

} // namespace mlir

#endif // MLIR_TOOLS_MLIR_PDLL_LSP_SERVER_MLIRPDLLLSPSERVERMAIN_H
