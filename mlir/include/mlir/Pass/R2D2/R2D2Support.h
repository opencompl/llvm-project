#ifndef MLIR_PASS_R2D2_R2D2SUPPORT_H_
#define MLIR_PASS_R2D2_R2D2SUPPORT_H_

#include "mlir/Pass/R2D2/R2D2.h"

namespace mlir {
namespace r2d2 {

using LocationQuery = SetVector<LocationOp>;

LogicalResult findAncestors(LocationQuery &out, LocationOp source,
                            unsigned maxDepth = UINT_MAX);
LogicalResult findDescendants(LocationQuery &out, LocationOp source,
                              unsigned maxDepth = UINT_MAX);
LogicalResult findRelatives(LocationQuery &out, LocationOp source,
                            unsigned maxDepth = UINT_MAX);

} // namespace r2d2
} // namespace mlir

#endif