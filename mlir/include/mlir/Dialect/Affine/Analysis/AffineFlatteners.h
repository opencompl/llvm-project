#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IntegerSet.h"

#ifndef MLIR_DIALECT_AFFINE_ANALYSIS_AFFINEFLATTENERS_H
#define MLIR_DIALECT_AFFINE_ANALYSIS_AFFINEFLATTENERS_H

namespace mlir {
namespace presburger {

inline LogicalResult addAffineForOpDomain(IntegerPolyhedron &cst, AffineForOp forOp) {
  FlatAffineValueConstraints tmp(cst);

  if (failed(tmp.addAffineForOpDomain(forOp)))
    return failure();

  cst = tmp;
  return success();
}

inline LogicalResult addAffineIfOpDomain(IntegerPolyhedron &cst, AffineIfOp ifOp) {
  FlatAffineValueConstraints tmp(cst);

  tmp.addAffineIfOpDomain(ifOp);

  cst = tmp;
  return success();
}

inline IntegerSet presburgerToIntegerSet(IntegerPolyhedron &cst, MLIRContext *ctx) {
  FlatAffineValueConstraints tmp(cst);
  return tmp.getAsIntegerSet(ctx);
}

} // namespace presburger
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_ANALYSIS_AFFINEFLATTENERS_H
