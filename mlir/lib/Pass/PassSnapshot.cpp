
#include "PassDetail.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/LocationSnapshot.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/R2D2/R2D2.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::detail;

namespace {

class PassSnapshotInstrumentation : public PassInstrumentation {
private:
  struct TraceBuilder {
    MLIRContext context{MLIRContext::Threading::DISABLED};
    OpBuilder builder{&context};
    r2d2::TraceOp trace;
    Value currentSnapshot;
    DenseMap<size_t, Value> prevHashToValue;
    DenseMap<size_t, Value> currHashToValue;

    static size_t hash(FileLineColLoc flcl) {
      return llvm::hash_combine(flcl.getFilename().strref(), flcl.getLine(),
                                flcl.getColumn());
    }

    explicit TraceBuilder(Operation *op) {
      auto sourceFlcl = dyn_cast<FileLineColLoc>(op->getLoc());
      assert(sourceFlcl);
      auto sourceFile = sourceFlcl.getFilename().strref();

      context.loadDialect<r2d2::R2D2Dialect>();
      trace = builder.create<r2d2::TraceOp>(
          FileLineColLoc::get(&context, sourceFile, 0, 0), sourceFile);

      auto &region = trace.getBody();
      auto &block = trace.getBody().emplaceBlock();
      currentSnapshot =
          region.addArgument(r2d2::SnapshotType::get(&context),
                             FileLineColLoc::get(&context, sourceFile, 0, 0));

      builder.setInsertionPoint(&block, block.begin());
      op->walk([this](Operation *operation) {
        if (auto flcl = dyn_cast<FileLineColLoc>(operation->getLoc())) {
          auto locOp = builder.create<r2d2::LocationOp>(
              FileLineColLoc::get(&context, flcl.getFilename(), flcl.getLine(),
                                  flcl.getColumn()),
              r2d2::LocationType::get(&context), currentSnapshot,
              flcl.getLine(), flcl.getColumn(), ValueRange{});
          currHashToValue.try_emplace(hash(flcl), locOp);
        } else
          assert(false && "unhandled case: location from source file is not a "
                          "FileLineColLoc.");
      });
    }

    void addPass(StringRef passName, StringRef snapshotFileName) {
      auto snapshot = builder.create<r2d2::PassOp>(
          FileLineColLoc::get(&context, snapshotFileName, 0, 0), passName,
          snapshotFileName, currentSnapshot);
      currentSnapshot = snapshot;
      std::swap(currHashToValue, prevHashToValue);
      currHashToValue.clear();
    }

    void addLocation(FileLineColLoc flcl, ArrayRef<FileLineColLoc> deps) {
      // need to search
      SmallVector<Value> vals(
          llvm::map_range(deps, [this](const FileLineColLoc &loc) -> Value {
            auto h = hash(loc);
            auto itr = prevHashToValue.find(h);
            assert(itr != prevHashToValue.end());
            return itr->second;
          }));

      auto locOp = builder.create<r2d2::LocationOp>(
          FileLineColLoc::get(&context, flcl.getFilename().strref(),
                              flcl.getLine(), flcl.getColumn()),
          currentSnapshot, flcl.getLine(), flcl.getColumn(), ValueRange{vals});
      currHashToValue.try_emplace(hash(flcl), locOp);
    }
  };

  /// Instrumentation hooks.
  void runBeforePipeline(std::optional<OperationName> name,
                         const PipelineParentInfo &parentInfo) override;
  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPipeline(std::optional<OperationName> name,
                        const PipelineParentInfo &parentInfo) override;

  unsigned passCount = 1;
  std::unique_ptr<TraceBuilder> traceBuilder;
};
} // namespace

namespace {

void PassSnapshotInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  if (!traceBuilder) {
    // first pass, snapshot the first op
    traceBuilder = std::make_unique<TraceBuilder>(op);
  }
}

void PassSnapshotInstrumentation::runAfterPass(Pass *pass, Operation *op) {

  const auto passId = passCount++;

  const auto filename = llvm::formatv("snap-pass{}.mlir", passId).str();
  const auto tagname = llvm::formatv("tag{}", passId).str();

  traceBuilder->addPass(pass->getName(), filename);
  (void)generateLocationsFromIR(filename, tagname, op,
                                OpPrintingFlags().enableDebugInfo(false));

  op->walk([this, passId](Operation *op) {
    if (auto fusedLoc = dyn_cast<FusedLoc>(op->getLoc())) {
      // establish a mapping

      std::optional<FileLineColLoc> thisLoc;
      SmallVector<FileLineColLoc> prevLocs;
      for (auto &subloc : llvm::reverse(fusedLoc.getLocations())) {
        // check if this is a tagged location
        // if true, this is an intermediate pass
        if (auto namedLoc = dyn_cast<NameLoc>(subloc)) {
          auto flcLoc = dyn_cast<FileLineColLoc>(namedLoc.getChildLoc());
          auto tag = namedLoc.getName().strref();

          unsigned locPass;
          if (tag.substr(3).getAsInteger(10, locPass))
            continue;

          if (locPass == passId) {
            // location is generated this pass, should always be last in list
            // since list is reversed, should always be visited first
            // also, we should never have more than one loc from this pass
            // assert that thisLoc is never initialized twice
            assert(!thisLoc);
            thisLoc = flcLoc;
          } else if (locPass + 1 == passId) {
            prevLocs.push_back(flcLoc);
          } else {
            // discard
          }
        }

        // this is an untagged location
        // this can only be the root MLIR file
        if (auto flcLoc = dyn_cast<FileLineColLoc>(subloc)) {
          // the first pass should capture the changes to the root mlir file
          if (passId == 1)
            prevLocs.push_back(flcLoc);

          // otherwise, process nothing
          continue;
        }
      }

      traceBuilder->addLocation(*thisLoc, prevLocs);
    }
  });

  llvm::errs() << "// ----\n";
  traceBuilder->trace.print(llvm::errs(),
                            OpPrintingFlags{}.enableDebugInfo(false));
}

void PassSnapshotInstrumentation::runBeforePipeline(
    std::optional<OperationName> name, const PipelineParentInfo &parentInfo) {
  llvm::errs() << "test runBefore\n";
}

void PassSnapshotInstrumentation::runAfterPipeline(
    std::optional<OperationName> name, const PipelineParentInfo &parentInfo) {
  if (traceBuilder) {
    llvm::errs() << traceBuilder->trace << "\n";
  }
}
} // namespace

void PassManager::enableSnapshot() {
  addInstrumentation(std::make_unique<PassSnapshotInstrumentation>());
}