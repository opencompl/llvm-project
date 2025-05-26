#include "mlir/Tools/mlir-r2d2-server/MlirR2d2ServerMain.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;

int main(int argc, char **argv) {
  return failed(MlirR2D2ServerMain(argc, argv));
}
