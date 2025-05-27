#include "mlir/Pass/R2D2/R2D2Support.h"
#include "llvm/ADT/SetVector.h"

#include <queue>

using namespace mlir;
using namespace mlir::r2d2;

namespace {
static LogicalResult traverseAncestors(LocationQuery &out, LocationOp source,
                                       unsigned maxDepth) {
  std::queue<std::pair<LocationOp, unsigned>> queue;
  llvm::DenseSet<Operation *> visited;
  queue.emplace(source, 0);
  visited.insert(source);

  while (queue.size() > 0) {
    auto top = queue.front();
    auto locOp = top.first;
    auto locDepth = top.second;
    queue.pop();

    out.insert(locOp);

    assert(top.second <= maxDepth);

    if (locDepth < maxDepth)
      for (auto elem : locOp.getArgs()) {
        auto *op = elem.getDefiningOp();
        if (!visited.contains(op)) {
          auto loc = dyn_cast<LocationOp>(op);

          visited.insert(op);
          queue.emplace(loc, locDepth + 1);
        }
      }
  }
  return success();
}

static LogicalResult traverseDescendants(LocationQuery &out, LocationOp source,
                                         unsigned maxDepth) {
  std::queue<std::pair<LocationOp, unsigned>> queue;
  llvm::DenseSet<Operation *> visited;
  queue.emplace(source, 0);

  while (queue.size() > 0) {
    auto top = queue.front();
    auto locOp = top.first;
    auto locDepth = top.second;
    queue.pop();

    assert(top.second <= maxDepth);

    if (locDepth < maxDepth)
      for (auto *op : locOp->getUsers()) {
        if (auto loc = dyn_cast<LocationOp>(op)) {
          if (!visited.contains(op)) {
            visited.insert(op);
            out.insert(loc);
            queue.emplace(loc, locDepth + 1);
          }
        }
      }
  }

  return success();
}
} // namespace

namespace mlir {
namespace r2d2 {
LogicalResult findAncestors(LocationQuery &out, LocationOp source,
                            unsigned maxDepth) {
  out.clear();
  return ::traverseAncestors(out, source, maxDepth);
}
LogicalResult findDescendants(LocationQuery &out, LocationOp source,
                              unsigned maxDepth) {
  out.clear();
  return ::traverseDescendants(out, source, maxDepth);
}
LogicalResult findRelatives(LocationQuery &out, LocationOp source,
                            unsigned maxDepth) {
  out.clear();
  return success(succeeded(::traverseAncestors(out, source, maxDepth)) &&
                 succeeded(::traverseDescendants(out, source, maxDepth)));
}
} // namespace r2d2
} // namespace mlir