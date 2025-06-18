#include "Protocol.h"

namespace mlir {
namespace r2d2 {

bool fromJSON(const llvm::json::Value &value, FileLine &result,
              llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("filename", result.filename) &&
         o.map("line", result.line);
}

llvm::json::Value toJSON(const FileLine &diag) {
  return llvm::json::Object{{"filename", diag.filename},
                            {"line", diag.line}};
}

bool fromJSON(const llvm::json::Value &value, LoadRequest &result,
              llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("str", result.str);
}

llvm::json::Value toJSON(const LoadSuccessResponse &diag) {
  return llvm::json::Object{{"status", "success"},
                            {"snapshots", diag.snapshots},
                            {"passes", diag.passes}};
}

llvm::json::Value toJSON(const LoadFailureResponse &diag) {
  return llvm::json::Object{{"status", "failure"},
                            {"errorMessage", diag.errorMessage}};
}

llvm::json::Value toJSON(const LoadResponse &diag) {
  return std::visit([](auto &&val) { return toJSON(val); }, diag);
}

bool fromJSON(const llvm::json::Value &value, TraceDirection &result,
              llvm::json::Path path) {
  if (std::optional<StringRef> str = value.getAsString()) {
    if (*str == "back") {
      result = TraceDirection::Backward;
      return true;
    }
    if (*str == "fwd") {
      result = TraceDirection::Forward;
      return true;
    }
  }
  return false;
}

bool fromJSON(const llvm::json::Value &value, TraceRequest &result,
              llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("source", result.source) &&
         o.map("traceDirection", result.traceDirection) &&
         o.map("maxDepth", result.maxDepth);
}

llvm::json::Value toJSON(const TraceRequest &diag) {
  return llvm::json::Object{{"source", diag.source},
                            {"maxDepth", diag.maxDepth}};
}

llvm::json::Value toJSON(const TraceResponse &diag) {
  return llvm::json::Object{{"locations", diag.locations}};
}
} // namespace r2d2
} // namespace mlir