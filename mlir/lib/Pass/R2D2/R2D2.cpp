#include "mlir/Pass/R2D2/R2D2.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Pass/R2D2/R2D2.cpp.inc"
#include "mlir/Pass/R2D2/R2D2Dialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Pass/R2D2/R2D2TypesGen.cpp.inc"
#define GET_OP_CLASSES
#include "mlir/Pass/R2D2/R2D2Ops.cpp.inc"

using namespace mlir;
using namespace mlir::r2d2;

void R2D2Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Pass/R2D2/R2D2Ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Pass/R2D2/R2D2TypesGen.cpp.inc"
      >();
}

void LocationOp::build(::mlir::OpBuilder &odsBuilder,
                       ::mlir::OperationState &odsState, Value snapshot,
                       unsigned line, unsigned column,
                       ::mlir::ValueRange args) {
  build(odsBuilder, odsState, LocationType::get(odsBuilder.getContext()),
        snapshot, line, column, args);
}

StringAttr LocationOp::getSnapshotFile() {
  auto pass = getSnapshot();
  if (auto passOp = dyn_cast<PassOp>(pass.getDefiningOp()))
    return passOp.getSnapshotAttr();

  // otherwise it is the source file
  auto traceOp = dyn_cast<TraceOp>(getOperation()->getParentOp());
  assert(traceOp);
  return traceOp.getSnapshotAttr();
}

FileLineColLoc LocationOp::getContainedLocation() {
  auto snapshot = getSnapshotFile();
  return FileLineColLoc::get(snapshot, getLine(), getColumn());
}

void PassOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState, StringAttr snapshot,
                   ::mlir::Value prev) {
  build(odsBuilder, odsState, SnapshotType::get(odsBuilder.getContext()),
        snapshot, prev);
}

void PassOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState, StringRef snapshot,
                   ::mlir::Value prev) {
  build(odsBuilder, odsState, SnapshotType::get(odsBuilder.getContext()),
        snapshot, prev);
}

::mlir::ParseResult TraceOp::parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result) {
  auto *ctx = parser.getContext();

  std::string fileName;
  if (parser.parseString(&fileName))
    return failure();

  OpAsmParser::Argument arg;
  if (parser.parseLParen())
    return failure();
  if (parser.parseArgument(arg, false, false))
    return failure();
  if (parser.parseRParen())
    return failure();

  result.addAttribute("snapshot", StringAttr::get(ctx, fileName));
  arg.type = r2d2::SnapshotType::get(ctx);

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, arg))
    return failure();

  return success();
}

void TraceOp::print(::mlir::OpAsmPrinter &p) {
  auto snapshot = getSnapshotAttr();
  p << " ";
  p.printAttribute(snapshot);
  p << " ";

  auto &body = getRegion();
  p << "(";
  auto sourceArg = body.getArgument(0);
  p.printRegionArgument(sourceArg, {}, true);
  p << ")";

  p.printRegion(getRegion(), false);
}
