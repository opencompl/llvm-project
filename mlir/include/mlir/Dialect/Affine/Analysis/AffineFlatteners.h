#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IntegerSet.h"

namespace mlir {
namespace presburger {

LogicalResult addAffineForOpDomain(IntegerPolyhedron &cst, AffineForOp forOp) {
  FlatAffineValueConstraints tmp(cst);

  if (failed(tmp.addAffineForOpDomain(forOp)))
    return failure();

  cst = tmp;
  return success();
}

LogicalResult addAffineIfOpDomain(IntegerPolyhedron &cst, AffineIfOp ifOp) {
  FlatAffineValueConstraints tmp(cst);

  tmp.addAffineIfOpDomain(ifOp);

  cst = tmp;
  return success();
}

IntegerSet presburgerToIntegerSet(IntegerPolyhedron &cst, MLIRContext *ctx) {
  FlatAffineValueConstraints tmp(cst);
  return tmp.getAsIntegerSet(ctx);
}

} // namespace presburger
} // namespace mlir

