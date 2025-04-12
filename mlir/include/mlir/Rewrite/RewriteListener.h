
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Cursor {
public:
  struct OpAtom {
    std::ptrdiff_t regionIndex;
    std::ptrdiff_t blockIndex;
    std::ptrdiff_t opIndex;
  };

  explicit Cursor(Operation *op);

  Operation *apply(ModuleOp *root) const;
  llvm::ArrayRef<OpAtom> getAtoms() const;

private:
  llvm::SmallVector<OpAtom> atoms;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Cursor &cursor);

static inline ModuleOp getModule(Operation *op) {
  auto *prev = op->getParentOp();
  while (prev) {
    op = prev;
    prev = op->getParentOp();
  };
  return dyn_cast<ModuleOp>(op);
}

class RewriteRecord {
public:
  using RecordSet = llvm::SmallVector<Cursor, 10>;
  RecordSet removed, added;
  const Pattern *pattern{};
  ModuleOp module;
};

class PassRewriteRecorder : public RewriterBase::Listener {
  std::vector<RewriteRecord> records;
  std::optional<RewriteRecord> currentRecord;

public:
  RewriteRecord::RecordSet dce;

public:
  PassRewriteRecorder() = default;

  llvm::ArrayRef<RewriteRecord> getRecords() const { return records; }

  void notifyOperationInserted(Operation *op,
                               RewriterBase::InsertPoint previous) override {
    llvm::errs() << "//---  op inserted, mlir state:\n";
    assert(currentRecord);
    currentRecord->added.emplace_back(Cursor{op});
  }

  void notifyOperationReplaced(Operation *op, Operation *newOp) override {
    llvm::errs() << "//---  op replaced\n";
    currentRecord->removed.emplace_back(Cursor{op});
    currentRecord->added.emplace_back(Cursor{newOp});
  }

  void notifyOperationErased(Operation *op) override {
    llvm::errs() << "//---  op erased\n";

    // may not be recording a pattern, due to DCE
    if (currentRecord)
      currentRecord->removed.emplace_back(Cursor{op});
    else
      dce.emplace_back(Cursor{op});
  }

  void notifyPatternBegin(const Pattern &pattern, Operation *op) override {
    llvm::errs() << "//---  pat begin\n";
    llvm::errs() << getModule(op) << "\n";
    assert(!currentRecord);
    auto &curr = currentRecord.emplace();
    curr.pattern = &pattern;
    curr.module = getModule(op);
  }

  void notifyPatternEnd(const Pattern &pattern, LogicalResult status) override {
    assert(currentRecord && currentRecord->pattern == &pattern);
    llvm::errs() << "//---  pat end\n";
    llvm::errs() << currentRecord->module << "\n";
    records.emplace_back(std::move(*currentRecord));
    currentRecord.reset();
  }

  void clear() {
    records.clear();
    currentRecord.reset();
  }
};
} // namespace mlir