#include "R2D2Server.h"
#include "Protocol.h"
#include "mlir/Pass/R2D2/R2D2Support.h"
#include "mlir/Tools/lsp-server-support/Transport.h"

namespace mlir {
namespace r2d2 {
using namespace lsp;

struct R2D2ServerForwarder {
public:
  R2D2ServerForwarder(R2D2Server &server) : r2d2{server} {}

  void onInitialize(const NoParams &params, Callback<llvm::json::Value> reply);
  void onShutdown(const NoParams &params, Callback<std::nullptr_t> reply);

  R2D2Server &r2d2;
  bool shutdownRequestReceived = false;
};

void R2D2ServerForwarder::onInitialize(const NoParams &params,
                                       Callback<llvm::json::Value> reply) {
  reply(llvm::json::Value{});
}
void R2D2ServerForwarder::onShutdown(const NoParams &params,
                                     Callback<std::nullptr_t> reply) {
  shutdownRequestReceived = true;
  reply(nullptr);
}

llvm::LogicalResult runR2D2Server(R2D2Server &server,
                                  JSONTransport &transport) {
  MessageHandler messageHandler{transport};
  R2D2ServerForwarder forwarder{server};

  messageHandler.method("initialize", &forwarder,
                        &R2D2ServerForwarder::onInitialize);
  messageHandler.method("shutdown", &forwarder,
                        &R2D2ServerForwarder::onShutdown);

  if (auto error = transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    llvm::consumeError(std::move(error));
    return failure();
  }

  return success(forwarder.shutdownRequestReceived);
}
} // namespace r2d2
} // namespace mlir
