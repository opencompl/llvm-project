//===- SimplifyAffineStructures.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to simplify if conditions in operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineFlatteners.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IntegerSet.h"

using namespace mlir;
using namespace presburger;

namespace {

struct CoalesceMemRefs : public CoalesceMemRefsBase<CoalesceMemRefs> {

  CoalesceMemRefs() : CoalesceMemRefsBase() {}

  void runOnOperation() override;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createCoalesceMemRefsPass() {
  return std::make_unique<CoalesceMemRefs>();
}

void CoalesceMemRefs::runOnOperation() {
  func::FuncOp op = getOperation();

  // Get all fill operations in this block.
  auto fillOps = op.getOps<linalg::FillOp>();

  // Check if the constant for the fill operations is same.
  auto firstOp = *fillOps.begin();
  auto firstConstant = firstOp.getInputOperand(0)->get();
  bool sameConstant = std::all_of(fillOps.begin(), fillOps.end(), [&](auto a) {
    return a.getInputOperand(0)->get() == firstConstant;
  });
  if (!sameConstant)
    return;

  // Collect all memrefs in fill defined by subviews.
  SmallVector<memref::SubViewOp, 2> subviews;
  for (auto fillOp : fillOps) {
    if (auto subview = dyn_cast<memref::SubViewOp>(
            fillOp.getOutputOperand(0)->get().getDefiningOp())) {
      subviews.push_back(subview);
    }
  }

  // There are some other checks that can be added such as checking if the
  // original memref of the subviews are same.
  for (auto subview : subviews) {
    auto parentMemref = subview->getResultTypes()[0].cast<MemRefType>();
    unsigned numDims = parentMemref.getRank();

    IntegerPolyhedron subviewSpace(PresburgerSpace::getSetSpace(numDims));
    for (unsigned i = 0; i < numDims; ++i) {
      int64_t offset = subview.getStaticOffset(i);
      int64_t size = subview.getStaticSize(i);
      int64_t stride = subview.getStaticStride(i);

      subviewSpace.addBound(IntegerPolyhedron::BoundType::LB, i, offset);
      subviewSpace.addBound(IntegerPolyhedron::BoundType::UB, i, offset + size);

      if (stride != 1) {
        // Add constraints for the stride.
        // (iv - lb) % step = 0 can be written as:
        // (iv - lb) - step * q = 0 where q = (iv - lb) / step.
        // Add local variable 'q' and add the above equality.
        // The first constraint is q = (iv - lb) floordiv step
        SmallVector<int64_t, 8> dividend(subviewSpace.getNumCols(), 0);
        dividend[i] = 1;
        dividend.back() -= offset;
        subviewSpace.addLocalFloorDiv(dividend, stride);

        // Second constraint: (iv - lb) - step * q = 0.
        SmallVector<int64_t, 8> eq(subviewSpace.getNumCols(), 0);
        eq[i] = 1;
        eq.back() -= offset;
        // For the local var just added above.
        eq[subviewSpace.getNumCols() - 2] = -stride;
        subviewSpace.addEquality(eq);
      }
    }

    subviewSpace.dump();
  }
}
