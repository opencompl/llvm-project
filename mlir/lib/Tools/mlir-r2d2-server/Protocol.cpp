#include "Protocol.h"

namespace mlir {
namespace r2d2 {

bool fromJSON(const llvm::json::Value &value, FileLineCol &result,
              llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("filename", result.filename) &&
         o.map("line", result.line) && o.map("column", result.column);
}

llvm::json::Value toJSON(const FileLineCol &diag) {
  return llvm::json::Object{{"filename", diag.filename},
                            {"line", diag.line},
                            {"column", diag.column}};
}

bool fromJSON(const llvm::json::Value &value, TraceRequest &result,
              llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("source", result.source) &&
         o.map("maxDepth", result.maxDepth);
}

llvm::json::Value toJSON(const TraceRequest &diag) {
  return llvm::json::Object{{"source", diag.source},
                            {"maxDepth", diag.maxDepth}};
}
} // namespace r2d2
} // namespace mlir