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
#include "mlir/IR/IntegerSet.h"

using namespace mlir;
using namespace presburger;

namespace {

struct SimplifyAffineIf : public SimplifyAffineIfBase<SimplifyAffineIf> {

  SimplifyAffineIf() : SimplifyAffineIfBase() {}

  void traverse(Operation *op, const PresburgerSet &cst);

  void traverse(Region &reg, const PresburgerSet &cst);

  void runOnOperation() override;
};

} // namespace

void SimplifyAffineIf::traverse(Region &region, const PresburgerSet &cst) {
  for (Block &block : region.getBlocks())
    for (Operation &op : block.getOperations())
      traverse(&op, cst);
}

void SimplifyAffineIf::traverse(Operation *op, const PresburgerSet &cst) {
  if (AffineForOp forOp = dyn_cast<AffineForOp>(*op)) {
    // TODO: Create iteration domain here and intersect with cst.

    // Create for conditions here.
    IntegerPolyhedron conditions(cst.getSpace());
    unsigned idPos = conditions.appendId(IdKind::SetDim);
    conditions.atValue(idPos) = forOp.getInductionVar();
    assert(succeeded(addAffineForOpDomain(conditions, forOp)));

    PresburgerSet copySet = cst;

    copySet.mergeIds(IdKind::SetDim, conditions);
    copySet.mergeIds(IdKind::Symbol, conditions);
    copySet = copySet.intersect(PresburgerSet(conditions));

    for (Region &region : op->getRegions())
      traverse(region, copySet);

  } else if (AffineIfOp ifOp = dyn_cast<AffineIfOp>(*op)) {
    // Create if constraints here.
    IntegerPolyhedron conditions(cst.getSpace());
    assert(succeeded(addAffineIfOpDomain(conditions, ifOp)));

    PresburgerSet copySet = cst;

    copySet.mergeIds(IdKind::SetDim, conditions);
    copySet.mergeIds(IdKind::Symbol, conditions);

    conditions.removeRedundantConstraints(copySet);

    SmallVector<Value, 8> values;
    conditions.getValues(0, conditions.getNumDimAndSymbolIds(), &values);

    ifOp.setConditional(presburgerToIntegerSet(conditions, ifOp.getContext()),
                        values);

    // Traverse then region with those constraints.
    traverse(ifOp.thenRegion(), copySet.intersect(PresburgerSet(conditions)));

    // Traverse else region with complement of constraints.
    traverse(ifOp.elseRegion(),
             copySet.intersect(PresburgerSet(conditions).complement()));

  } else {
    for (Region &region : op->getRegions())
      traverse(region, cst);
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createSimplifyAffineIfPass() {
  return std::make_unique<SimplifyAffineIf>();
}

void SimplifyAffineIf::runOnOperation() {
  func::FuncOp op = getOperation();
  PresburgerSet cst =
      PresburgerSet::getUniverse(PresburgerSpace::getSetSpace(0, 0, 0));

  traverse(op.getOperation(), cst);
}
