
#include "PassDetail.h"
#include "mlir/IR/LocationSnapshot.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::detail;

namespace {

class PassSnapshotInstrumentation : public PassInstrumentation {
private:
  /// Instrumentation hooks.
  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override;
};
} // namespace

namespace {
void PassSnapshotInstrumentation::runBeforePass(Pass *pass, Operation *op) {}
void PassSnapshotInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  (void)generateLocationsFromIR(
      Twine{"snapshot-"}.concat(pass->getName()).concat(".mlir").str(), op,
      OpPrintingFlags().enableDebugInfo());
}
void PassSnapshotInstrumentation::runAfterPassFailed(Pass *pass,
                                                     Operation *op) {}
} // namespace

void PassManager::enableSnapshot() {
  addInstrumentation(std::make_unique<PassSnapshotInstrumentation>());
}