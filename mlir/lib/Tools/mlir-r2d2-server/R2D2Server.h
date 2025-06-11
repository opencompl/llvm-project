#ifndef LIB_MLIR_TOOLS_MLIRR2D2SERVER_SERVER_H_
#define LIB_MLIR_TOOLS_MLIRR2D2SERVER_SERVER_H_

#include "Protocol.h"
#include "mlir/Pass/R2D2/R2D2Support.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>

namespace llvm {
struct LogicalResult;
}

namespace mlir {
namespace lsp {
class JSONTransport;
}

namespace r2d2 {
class R2D2Server {
public:
  llvm::Error loadR2D2File(llvm::StringRef r2d2);
  LocationOp findOp(llvm::StringRef source, unsigned line, unsigned col);
  std::optional<LocationQuery>
  findRelatives(LocationOp source, TraceDirection direction, unsigned maxDepth);

  std::vector<std::string> getSnapshots() const;

  R2D2Server();
  R2D2Server(R2D2Server &&) noexcept;
  R2D2Server &operator=(R2D2Server &&) noexcept;
  ~R2D2Server() noexcept;

private:
  struct impl;
  std::unique_ptr<impl> pimpl;
};
llvm::LogicalResult runR2D2Server(R2D2Server &server,
                                  lsp::JSONTransport &transport);
} // namespace r2d2
} // namespace mlir
#endif