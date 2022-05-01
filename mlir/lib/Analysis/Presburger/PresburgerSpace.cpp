//===- PresburgerSpace.cpp - MLIR PresburgerSpace Class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <cassert>

using namespace mlir;
using namespace presburger;

unsigned PresburgerSpace::getNumIdKind(IdKind kind) const {
  if (kind == IdKind::Domain)
    return getNumDomainIds();
  if (kind == IdKind::Range)
    return getNumRangeIds();
  if (kind == IdKind::Symbol)
    return getNumSymbolIds();
  if (kind == IdKind::Local)
    return numLocals;
  llvm_unreachable("IdKind does not exist!");
}

unsigned PresburgerSpace::getIdKindOffset(IdKind kind) const {
  if (kind == IdKind::Domain)
    return 0;
  if (kind == IdKind::Range)
    return getNumDomainIds();
  if (kind == IdKind::Symbol)
    return getNumDimIds();
  if (kind == IdKind::Local)
    return getNumDimAndSymbolIds();
  llvm_unreachable("IdKind does not exist!");
}

unsigned PresburgerSpace::getIdKindEnd(IdKind kind) const {
  return getIdKindOffset(kind) + getNumIdKind(kind);
}

unsigned PresburgerSpace::getIdKindOverlap(IdKind kind, unsigned idStart,
                                           unsigned idLimit) const {
  unsigned idRangeStart = getIdKindOffset(kind);
  unsigned idRangeEnd = getIdKindEnd(kind);

  // Compute number of elements in intersection of the ranges [idStart, idLimit)
  // and [idRangeStart, idRangeEnd).
  unsigned overlapStart = std::max(idStart, idRangeStart);
  unsigned overlapEnd = std::min(idLimit, idRangeEnd);

  if (overlapStart > overlapEnd)
    return 0;
  return overlapEnd - overlapStart;
}

IdKind PresburgerSpace::getIdKindAt(unsigned pos) const {
  assert(pos < getNumIds() && "`pos` should represent a valid id position");
  if (pos < getIdKindEnd(IdKind::Domain))
    return IdKind::Domain;
  if (pos < getIdKindEnd(IdKind::Range))
    return IdKind::Range;
  if (pos < getIdKindEnd(IdKind::Symbol))
    return IdKind::Symbol;
  if (pos < getIdKindEnd(IdKind::Local))
    return IdKind::Local;
  llvm_unreachable("`pos` should represent a valid id position");
}

unsigned PresburgerSpace::insertId(IdKind kind, unsigned pos, unsigned num) {
  assert(pos <= getNumIdKind(kind));

  unsigned absolutePos = getIdKindOffset(kind) + pos;

  if (kind == IdKind::Domain)
    numDomain += num;
  else if (kind == IdKind::Range)
    numRange += num;
  else if (kind == IdKind::Symbol)
    numSymbols += num;
  else
    numLocals += num;

  // Insert values for newly added variables.
  values.insert(values.begin() + absolutePos, num, None);

  assert(isConsistent() && "Space must be consistent.");

  return absolutePos;
}

void PresburgerSpace::removeIdRange(IdKind kind, unsigned idStart,
                                    unsigned idLimit) {
  assert(idLimit <= getNumIdKind(kind) && "invalid id limit");

  if (idStart >= idLimit)
    return;

  unsigned numIdsEliminated = idLimit - idStart;
  if (kind == IdKind::Domain)
    numDomain -= numIdsEliminated;
  else if (kind == IdKind::Range)
    numRange -= numIdsEliminated;
  else if (kind == IdKind::Symbol)
    numSymbols -= numIdsEliminated;
  else
    numLocals -= numIdsEliminated;

  // Remove values for removed variables.
  unsigned offset = getIdKindOffset(kind);
  values.erase(values.begin() + offset + idStart,
               values.begin() + offset + idLimit);

  assert(isConsistent() && "Space must be consistent.");
}

bool PresburgerSpace::isConsistent() const {
  return values.size() == getNumIds();
}

bool PresburgerSpace::isCompatible(const PresburgerSpace &other) const {
  assert(isConsistent() && "Space must be consistent.");
  return getNumDomainIds() == other.getNumDomainIds() &&
         getNumRangeIds() == other.getNumRangeIds() &&
         getNumSymbolIds() == other.getNumSymbolIds();
}

bool PresburgerSpace::isEqual(const PresburgerSpace &other) const {
  assert(isConsistent() && "Space must be consistent.");
  return isCompatible(other) && getNumLocalIds() == other.getNumLocalIds();
}

void PresburgerSpace::setDimSymbolSeparation(unsigned newSymbolCount) {
  assert(isConsistent() && "Space must be consistent.");
  assert(newSymbolCount <= getNumDimAndSymbolIds() &&
         "invalid separation position");
  // No need to modify `values`, since only the seperation is changed.
  numRange = numRange + numSymbols - newSymbolCount;
  numSymbols = newSymbolCount;
}

void PresburgerSpace::print(llvm::raw_ostream &os) const {
  os << "Domain: " << getNumDomainIds() << ", "
     << "Range: " << getNumRangeIds() << ", "
     << "Symbols: " << getNumSymbolIds() << ", "
     << "Locals: " << getNumLocalIds() << "\n";
}

void PresburgerSpace::dump() const { print(llvm::errs()); }

/// Checks if the SSA values associated with `cst`'s identifiers in range
/// [start, end) are unique.
bool LLVM_ATTRIBUTE_UNUSED PresburgerSpace::areIdsUnique(unsigned start,
                                                         unsigned end) {

  assert(start <= getNumIds() && "Start position out of bounds");
  assert(end <= getNumIds() && "End position out of bounds");

  if (start >= end)
    return true;

  SmallPtrSet<Value, 8> uniqueIds;
  ArrayRef<Optional<Value>> maybeValues = getMaybeValues();
  for (Optional<Value> val : maybeValues) {
    if (val.hasValue() && !uniqueIds.insert(val.getValue()).second)
      return false;
  }
  return true;
}

/// Checks if the SSA values associated with `cst`'s identifiers are unique.
bool LLVM_ATTRIBUTE_UNUSED PresburgerSpace::areIdsUnique() {
  return areIdsUnique(0, getNumIds());
}

/// Checks if the SSA values associated with `cst`'s identifiers of kind `kind`
/// are unique.
bool LLVM_ATTRIBUTE_UNUSED PresburgerSpace::areIdsUnique(IdKind kind) {

  switch (kind) {
  case IdKind::Domain:
    return areIdsUnique(getIdKindOffset(IdKind::Domain),
                        getIdKindEnd(IdKind::Domain));
  case IdKind::Range:
    return areIdsUnique(getIdKindOffset(IdKind::Range),
                        getIdKindEnd(IdKind::Range));
  case IdKind::Symbol:
    return areIdsUnique(getIdKindOffset(IdKind::Symbol),
                        getIdKindEnd(IdKind::Symbol));
  case IdKind::Local:
    return areIdsUnique(getIdKindOffset(IdKind::Local),
                        getIdKindEnd(IdKind::Local));
  }

  llvm_unreachable("Unexpected IdKind");
}

/// Checks if two constraint systems are in the same space, i.e., if they are
/// associated with the same set of identifiers, appearing in the same order.
static bool areIdsAligned(const PresburgerSpace &a, const PresburgerSpace &b) {
  return a.getNumDimIds() == b.getNumDimIds() &&
         a.getNumSymbolIds() == b.getNumSymbolIds() &&
         a.getNumIds() == b.getNumIds() &&
         a.getMaybeValues().equals(b.getMaybeValues());
}

/// Calls areIdsAligned to check if two constraint systems have the same set
/// of identifiers in the same order.
bool PresburgerSpace::areIdsAlignedWithOther(const PresburgerSpace &other) const {
  return areIdsAligned(*this, other);
}
