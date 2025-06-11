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

struct LoadRequest {
  std::string str;
};
/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, LoadRequest &result,
              llvm::json::Path path);

struct LoadSuccessResponse {
  std::vector<std::string> passes;
  std::vector<std::string> snapshots;
};

llvm::json::Value toJSON(const LoadSuccessResponse &diag);

struct LoadFailureResponse {
  std::string errorMessage;
};

llvm::json::Value toJSON(const LoadFailureResponse &diag);

using LoadResponse = std::variant<LoadSuccessResponse, LoadFailureResponse>;
llvm::json::Value toJSON(const LoadResponse &diag);

enum class TraceDirection { Backward, Forward };
/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, TraceDirection &result,
              llvm::json::Path path);

struct TraceRequest {
  FileLineCol source;
  TraceDirection traceDirection;
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
template <typename T>
llvm::json::Value toJSON(const std::vector<T> &diag) {
  auto a = llvm::json::Array();
  a.reserve(diag.size());
  for (auto &elem : diag)
    a.push_back(toJSON(elem));
  return a;
}

llvm::json::Value toJSON(const TraceResponse &diag);

} // namespace r2d2
} // namespace mlir

#endif