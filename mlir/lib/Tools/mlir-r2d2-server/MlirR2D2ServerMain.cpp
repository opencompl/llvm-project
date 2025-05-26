
#include "mlir/Tools/mlir-r2d2-server/MlirR2d2ServerMain.h"
#include "R2D2Server.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/Program.h"
using namespace mlir;
using namespace mlir::lsp;

LogicalResult mlir::MlirR2D2ServerMain(int argc, char **argv) {

  auto logLevel = Logger::Level::Debug;
  Logger::setLogLevel(logLevel);
  if (auto err = llvm::sys::ChangeStdinToBinary())
    return failure();

  JSONTransport transport(stdin, llvm::outs());
  r2d2::R2D2Server server;

  return r2d2::runR2D2Server(server, transport);
}