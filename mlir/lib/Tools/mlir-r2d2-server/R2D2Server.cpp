#include "R2D2Server.h"
#include "Protocol.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/R2D2/R2D2Support.h"
#include "mlir/Tools/lsp-server-support/Transport.h"

namespace mlir {
namespace r2d2 {
using namespace lsp;

namespace {
FileLineCol toFlc(LocationOp loc) {
  auto snapshotFile = loc.getSnapshotFile();
  return FileLineCol{snapshotFile.str(), loc.getLine(), loc.getColumn()};
}
} // namespace

struct R2D2Server::impl {
  mlir::MLIRContext ctx;
  llvm::SourceMgr sourceMgr;
  OwningOpRef<ModuleOp> module;
  TraceOp trace;
  llvm::StringMap<Value> snapshotCache;

  impl() { ctx.loadDialect<r2d2::R2D2Dialect>(); }
};

R2D2Server::R2D2Server() : pimpl{new impl} {}
R2D2Server::R2D2Server(R2D2Server &&) noexcept = default;
R2D2Server &R2D2Server::operator=(R2D2Server &&) noexcept = default;
R2D2Server::~R2D2Server() noexcept = default;

llvm::LogicalResult R2D2Server::loadR2D2File(llvm::StringRef r2d2) {
  auto *ctx = &pimpl->ctx;
  auto &sourceMgr = pimpl->sourceMgr;
  auto &trace = pimpl->trace;
  auto &module = pimpl->module;

  auto src = llvm::MemoryBuffer::getMemBuffer(r2d2);
  sourceMgr.AddNewSourceBuffer(std::move(src), SMLoc());
  module = parseSourceFile<ModuleOp>(sourceMgr, ctx);
  if (!module)
    return failure();

  if (auto traces = module->getOps<TraceOp>(); !traces.empty())
    trace = *traces.begin();
  else
    return failure();

  std::string output;
  llvm::raw_string_ostream stringStream(output);
  trace.print(stringStream);

  Logger::info("loaded r2d2: {}", output);

  auto &snapshotCache = pimpl->snapshotCache;
  snapshotCache[trace.getSnapshot()] = trace.getBody().getArgument(0);

  for (auto pass : trace.getOps<PassOp>())
    snapshotCache[pass.getSnapshot()] = pass;

  for (auto &&[name, val] : snapshotCache)
    Logger::info("detected snapshot {0} at {1}", name, val);

  return llvm::success(pimpl->trace);
}

LocationOp R2D2Server::findOp(llvm::StringRef source, unsigned line,
                              unsigned col) {
  auto &snapshotCache = pimpl->snapshotCache;
  if (auto itr = snapshotCache.find(source); itr != snapshotCache.end()) {
    for (auto *user : itr->second.getUsers()) {
      if (auto loc = dyn_cast<LocationOp>(user)) {
        if (loc.getLine() == line && loc.getColumn() == col)
          return loc;
      }
    }
  }
  return {};
}

std::optional<LocationQuery> R2D2Server::findRelatives(LocationOp source,
                                                       TraceDirection direction,
                                                       unsigned maxDepth) {
  LocationQuery query;
  switch (direction) {
  case TraceDirection::Backward:
    if (succeeded(findAncestors(query, source, maxDepth)))
      return std::move(query);
    else
      return std::nullopt;
  case TraceDirection::Forward:
    if (succeeded(findDescendants(query, source, maxDepth)))
      return std::move(query);
    else
      return std::nullopt;
  }
}

struct R2D2ServerForwarder {
public:
  R2D2ServerForwarder(R2D2Server &server) : r2d2{server} {}

  void onInitialize(const NoParams &params, Callback<std::string> reply);
  void onShutdown(const NoParams &params, Callback<std::nullptr_t> reply);

  void onR2D2LoadRequest(const LoadRequest &params,
                         Callback<std::string> reply);
  void onR2D2TraceRequest(const TraceRequest &params,
                          Callback<TraceResponse> reply);

  R2D2Server &r2d2;
  bool shutdownRequestReceived = false;
};

void R2D2ServerForwarder::onInitialize(const NoParams &params,
                                       Callback<std::string> reply) {
  reply("initialized");
}

void R2D2ServerForwarder::onShutdown(const NoParams &params,
                                     Callback<std::nullptr_t> reply) {
  shutdownRequestReceived = true;
  reply(nullptr);
}

void R2D2ServerForwarder::onR2D2LoadRequest(const LoadRequest &params,
                                            Callback<std::string> reply) {
  auto res = r2d2.loadR2D2File(params);
  reply(succeeded(res) ? "success" : "failed");
}

void R2D2ServerForwarder::onR2D2TraceRequest(const TraceRequest &params,
                                             Callback<TraceResponse> reply) {
  auto srcLoc = r2d2.findOp(params.source.filename, params.source.line,
                            params.source.column);
  if (!srcLoc)
    return;

  Logger::info("found {0}", srcLoc);

  auto query =
      r2d2.findRelatives(srcLoc, params.traceDirection, params.maxDepth);

  if (query) {
    TraceResponse response;
    response.locations.reserve(query->size());
    for (auto loc : *query)
      response.locations.emplace_back(toFlc(loc));

    reply(response);
  }
}

llvm::LogicalResult runR2D2Server(R2D2Server &server,
                                  JSONTransport &transport) {
  R2D2ServerForwarder forwarder{server};
  MessageHandler messageHandler{transport};

  messageHandler.method("initialize", &forwarder,
                        &R2D2ServerForwarder::onInitialize);
  messageHandler.method("exit", &forwarder, &R2D2ServerForwarder::onShutdown);

  messageHandler.method("r2d2/load", &forwarder,
                        &R2D2ServerForwarder::onR2D2LoadRequest);
  messageHandler.method("r2d2/trace", &forwarder,
                        &R2D2ServerForwarder::onR2D2TraceRequest);

  if (auto error = transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    llvm::consumeError(std::move(error));
    return failure();
  }

  return success(forwarder.shutdownRequestReceived);
}
} // namespace r2d2
} // namespace mlir
