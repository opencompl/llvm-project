#include "mlir/Rewrite/RewriteListener.h"

namespace mlir {

// op -> region -> block -> op ...
Cursor::Cursor(Operation *op) {
  while (!isa<ModuleOp>(op)) {
    auto block = op->getBlock();
    auto indexInBlock = std::distance(
        block->begin(),
        std::find_if(block->begin(), block->end(),
                     [op](const Operation &check) { return &check == op; }));
    auto region = block->getParent();
    auto indexInRegion = std::distance(
        region->begin(),
        std::find_if(region->begin(), region->end(),
                     [block](const Block &check) { return &check == block; }));
    op = region->getParentOp();
    auto opRegions = op->getRegions();
    auto indexInOp = std::distance(
        opRegions.begin(),
        std::find_if(opRegions.begin(), opRegions.end(),
                     [region](auto &check) { return &check == region; }));

    atoms.push_back(OpAtom{.regionIndex = indexInOp,
                           .blockIndex = indexInRegion,
                           .opIndex = indexInBlock});
  }
  std::reverse(atoms.begin(), atoms.end());
}

Operation *Cursor::apply(ModuleOp *root) const {
  auto *op = root->getOperation();
  for (const auto &atom : atoms) {
    auto &region = op->getRegion(atom.regionIndex);
    auto block = std::next(region.getBlocks().begin(), atom.blockIndex);
    op = std::addressof(
        *std::next(block->getOperations().begin(), atom.opIndex));
  }
  return op;
}

llvm::ArrayRef<Cursor::OpAtom> Cursor::getAtoms() const { return atoms; }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Cursor &cursor) {
  for (const auto &atom : cursor.getAtoms()) {
    os << atom.regionIndex << "." << atom.blockIndex << "." << atom.opIndex
       << "-";
  }
  return os;
};
} // namespace mlir