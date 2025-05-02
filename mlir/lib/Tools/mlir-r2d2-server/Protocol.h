#ifndef LIB_MLIR_TOOLS_MLIRR2D2SERVER_TRANSPORT_H_
#define LIB_MLIR_TOOLS_MLIRR2D2SERVER_TRANSPORT_H_

#include "mlir/IR/Location.h"
#include "llvm/Support/JSON.h"
#include <vector>

namespace mlir {
namespace r2d2 {

struct FileLineCol {
  std::string filename;
  unsigned line;
  unsigned column;
};
/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, FileLineCol &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const FileLineCol &diag);

struct TraceRequest {
  FileLineCol source;
  unsigned maxDepth = UINT_MAX;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, TraceRequest &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const TraceRequest &diag);

struct TraceResponse {
  std::vector<FileLineCol> locations;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, TraceResponse &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const TraceResponse &diag);

} // namespace r2d2
} // namespace mlir

#endif