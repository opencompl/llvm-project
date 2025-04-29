#include "mlir/Pass/MLDR/MLDR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Pass/MLDR/MLDR.cpp.inc"
#include "mlir/Pass/MLDR/MLDRDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Pass/MLDR/MLDRTypesGen.cpp.inc"
#define GET_OP_CLASSES
#include "mlir/Pass/MLDR/MLDROps.cpp.inc"

using namespace mlir;
using namespace mlir::mldr;

void MLDRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Pass/MLDR/MLDROps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Pass/MLDR/MLDRTypesGen.cpp.inc"
      >();
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
  arg.type = mldr::SnapshotType::get(ctx);

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
