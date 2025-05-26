#ifndef LIB_MLIR_TOOLS_MLIRR2D2SERVER_SERVER_H_
#define LIB_MLIR_TOOLS_MLIRR2D2SERVER_SERVER_H_

namespace llvm {
struct LogicalResult;
}

namespace mlir {
namespace lsp {
class JSONTransport;
}

namespace r2d2 {
class R2D2Server {};
llvm::LogicalResult runR2D2Server(R2D2Server &server,
                                  lsp::JSONTransport &transport);
} // namespace r2d2
} // namespace mlir
#endif