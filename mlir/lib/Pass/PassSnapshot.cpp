
#include "PassDetail.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/LocationSnapshot.h"
#include "mlir/Pass/MLDR/MLDR.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::detail;

namespace {

class PassSnapshotInstrumentation : public PassInstrumentation {
private:
  struct Trace {
    MLIRContext context{MLIRContext::Threading::DISABLED};
    OpBuilder builder{&context};
    mldr::TraceOp trace;
    Value currentSnapshot;
    DenseMap<size_t, Value> prevHashToValue;
    DenseMap<size_t, Value> currHashToValue;

    static size_t hash(FileLineColLoc flcl) {
      return llvm::hash_combine(flcl.getFilename().strref(), flcl.getLine(),
                                flcl.getColumn());
    }

    explicit Trace(Operation *op) {
      auto flcLoc = dyn_cast<FileLineColLoc>(op->getLoc());
      assert(flcLoc);
      auto source = flcLoc.getFilename().strref();

      context.loadDialect<mldr::MLDRDialect>();
      llvm::errs() << llvm::join(llvm::map_range(
                                     builder.getContext()->getLoadedDialects(),
                                     [](Dialect *dialect) -> StringRef {
                                       return dialect->getNamespace();
                                     }),
                                 ",")
                   << "\n";

      trace = builder.create<mldr::TraceOp>(
          FileLineColLoc::get(&context, source, 0, 0), source.str());

      auto &region = trace.getBody();
      auto &block = trace.getBody().emplaceBlock();
      currentSnapshot =
          region.addArgument(mldr::SnapshotType::get(&context),
                             FileLineColLoc::get(&context, source, 0, 0));

      builder.setInsertionPoint(&block, block.begin());
      op->walk([this](Operation *operation) {
        auto flcl = dyn_cast<FileLineColLoc>(operation->getLoc());
        if (flcl) {
          auto locOp = builder.create<mldr::LocationOp>(
              FileLineColLoc::get(&context, flcl.getFilename(), flcl.getLine(),
                                  flcl.getColumn()),
              mldr::LocationType::get(&context), currentSnapshot,
              flcl.getLine(), flcl.getColumn(), ValueRange{});
          currHashToValue.try_emplace(hash(flcl), locOp);
        } else {
          llvm::errs() << *operation << "\n";
        }
      });

      llvm::errs() << "// ----- start dump of debug\n";
      trace.print(llvm::errs(), OpPrintingFlags().enableDebugInfo(false));
      llvm::errs() << "// ----- end dump of debug\n";
    }

    void addPass(StringRef snapshotFileName) {
      auto snapshot = builder.create<mldr::PassOp>(
          FileLineColLoc::get(&context, snapshotFileName, 0, 0),
          mldr::SnapshotType::get(&context), snapshotFileName, currentSnapshot);
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

      auto locOp = builder.create<mldr::LocationOp>(
          FileLineColLoc::get(&context, flcl.getFilename().strref(),
                              flcl.getLine(), flcl.getColumn()),
          mldr::LocationType::get(&context), currentSnapshot, flcl.getLine(),
          flcl.getColumn(), ValueRange{vals});
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
  std::vector<StringRef> passNames;
  std::unique_ptr<Trace> trace;
};
} // namespace

namespace {

void PassSnapshotInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  if (!trace) {
    // first pass, snapshot the first op
    trace = std::make_unique<Trace>(op);
  }
}

void PassSnapshotInstrumentation::runAfterPass(Pass *pass, Operation *op) {

  const auto passId = passCount++;
  passNames.push_back(pass->getName());

  const auto filename = llvm::formatv("snap-pass{}.mlir", passId).str();
  const auto tagname = llvm::formatv("tag{}", passId).str();

  trace->addPass(filename);
  (void)generateLocationsFromIR(filename, tagname, op, OpPrintingFlags());

  llvm::errs() << "// Start dump of " << op->getName() << "\n";

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
          // process nothing
          if (passId == 1) {
            prevLocs.push_back(flcLoc);
          }
          continue;
        }
      }

      trace->addLocation(*thisLoc, prevLocs);
    }
  });

  trace->trace.print(llvm::errs(), OpPrintingFlags().enableDebugInfo(false));
  llvm::errs() << "// End dump\n";
}

void PassSnapshotInstrumentation::runBeforePipeline(
    std::optional<OperationName> name, const PipelineParentInfo &parentInfo) {}

void PassSnapshotInstrumentation::runAfterPipeline(
    std::optional<OperationName> name, const PipelineParentInfo &parentInfo) {}
} // namespace

void PassManager::enableSnapshot() {
  addInstrumentation(std::make_unique<PassSnapshotInstrumentation>());
}