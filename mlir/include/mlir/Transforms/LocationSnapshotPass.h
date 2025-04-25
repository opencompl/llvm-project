
#ifndef MLIR_TRANSFORMS_LOCATIONSNAPSHOTPASS_H
#define MLIR_TRANSFORMS_LOCATIONSNAPSHOTPASS_H

#include "mlir/IR/LocationSnapshot.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir {
#define GEN_PASS_DECL_LOCATIONSNAPSHOT
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#endif // MLIR_TRANSFORMS_LOCATIONSNAPSHOTPASS_H
