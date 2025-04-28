
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

  int passCount = 0;
  std::vector<StringRef> passNames;
};
} // namespace

namespace {
void PassSnapshotInstrumentation::runBeforePass(Pass *pass, Operation *op) {}

void PassSnapshotInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  const auto passId = passCount++;
  passNames.push_back(pass->getName());

  const auto filename = llvm::formatv("snap-pass{}.mlir", passId).str();
  const auto tagname = llvm::formatv("tag{}", passId).str();

  (void)generateLocationsFromIR(filename, tagname, op,
                                OpPrintingFlags().enableDebugInfo());
  llvm::errs() << "// Start dump of " << op->getName() << "\n";
  // op->getLoc()->walk([](Location loc) -> WalkResult {
  //   llvm::errs() << loc << "\n";
  //   return WalkResult::advance();
  // });
  llvm::errs() << *op << "\n";
  llvm::errs() << "// End dump\n";
}

void PassSnapshotInstrumentation::runAfterPassFailed(Pass *pass,
                                                     Operation *op) {}
} // namespace

void PassManager::enableSnapshot() {
  addInstrumentation(std::make_unique<PassSnapshotInstrumentation>());
}