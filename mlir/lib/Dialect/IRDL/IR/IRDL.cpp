//===- IRDL.cpp - IRDL dialect ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLSymbols.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::irdl;

//===----------------------------------------------------------------------===//
// IRDL dialect.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.cpp.inc"

#include "mlir/Dialect/IRDL/IR/IRDLDialect.cpp.inc"

void IRDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/IRDL/IR/IRDLOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/IRDL/IR/IRDLTypesGen.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Parsing/Printing
//===----------------------------------------------------------------------===//

/// Parse a region, and add a single block if the region is empty.
/// If no region is parsed, create a new region with a single empty block.
static ParseResult parseSingleBlockRegion(OpAsmParser &p, Region &region) {
  auto regionParseRes = p.parseOptionalRegion(region);
  if (regionParseRes.has_value() && failed(regionParseRes.value()))
    return failure();

  // If the region is empty, add a single empty block.
  if (region.empty())
    region.push_back(new Block());

  return success();
}

static void printSingleBlockRegion(OpAsmPrinter &p, Operation *op,
                                   Region &region) {
  if (!region.getBlocks().front().empty())
    p.printRegion(region);
}

LogicalResult DialectOp::verify() {
  if (!Dialect::isValidNamespace(getName()))
    return emitOpError("invalid dialect name");
  return success();
}

LogicalResult OperandsOp::verify() {
  size_t numNames = getOperandNames().size();
  size_t numVariadicities = getVariadicity().size();
  size_t numOperands = getNumOperands();

  if (numOperands != numVariadicities)
    return emitOpError()
           << "the number of operands and their variadicities must be "
              "the same, but got "
           << numOperands << " and " << numVariadicities << " respectively";

  if (numNames != numOperands)
    return emitOpError()
           << "the number of operand names and their constraints must be "
              "the same, but got "
           << numNames << " and " << numOperands << " respectively";

  return success();
}

LogicalResult ResultsOp::verify() {
  size_t numNames = getResultNames().size();
  size_t numVariadicities = getVariadicity().size();
  size_t numResults = this->getNumOperands();

  if (numResults != numVariadicities)
    return emitOpError()
           << "the number of results and their variadicities must be "
              "the same, but got "
           << numResults << " and " << numVariadicities << " respectively";

  if (numNames != numResults)
    return emitOpError()
           << "the number of result names and their constraints must be "
              "the same, but got "
           << numNames << " and " << numResults << " respectively";

  return success();
}

LogicalResult AttributesOp::verify() {
  size_t numNames = getAttributeValueNames().size();
  size_t numAttrs = getAttributeValues().size();
  size_t numVariadicities = getVariadicity().size();

  if (numVariadicities != numAttrs)
    return emitOpError()
           << "the number of attributes and their variadicities must be "
              "the same, but got "
           << numAttrs << " and " << numVariadicities << " respectively";

  if (numNames != numAttrs)
    return emitOpError()
           << "the number of attribute names and their constraints must be "
              "the same, but got "
           << numNames << " and " << numAttrs << " respectively";

  return success();
}

LogicalResult RegionsOp::verify() {
  size_t numNames = getRegionNames().size();
  size_t numRegions = getArgs().size();
  size_t numVariadicities = getVariadicity().size();
  if (numVariadicities != numRegions)
    return emitOpError()
           << "the number of regions and their variadicities must be "
              "the same, but got "
           << numRegions << " and " << numVariadicities << " respectively";

  if (numNames != numRegions)
    return emitOpError()
           << "the number of attribute names and their constraints must be "
              "the same, but got "
           << numNames << " and " << numRegions << " respectively";

  return success();
}

LogicalResult BaseOp::verify() {
  std::optional<StringRef> baseName = getBaseName();
  std::optional<SymbolRefAttr> baseRef = getBaseRef();
  if (baseName.has_value() == baseRef.has_value())
    return emitOpError() << "the base type or attribute should be specified by "
                            "either a name or a reference";

  if (baseName &&
      (baseName->empty() || ((*baseName)[0] != '!' && (*baseName)[0] != '#')))
    return emitOpError() << "the base type or attribute name should start with "
                            "'!' or '#'";

  return success();
}

/// Finds whether the provided symbol is an IRDL type or attribute definition.
/// The source operation must be within a DialectOp.
static LogicalResult
checkSymbolIsTypeOrAttribute(SymbolTableCollection &symbolTable,
                             Operation *source, SymbolRefAttr symbol) {
  Operation *targetOp =
      irdl::lookupSymbolNearDialect(symbolTable, source, symbol);

  if (!targetOp)
    return source->emitOpError() << "symbol '" << symbol << "' not found";

  if (!isa<TypeOp, AttributeOp>(targetOp))
    return source->emitOpError() << "symbol '" << symbol
                                 << "' does not refer to a type or attribute "
                                    "definition (refers to '"
                                 << targetOp->getName() << "')";

  return success();
}

LogicalResult BaseOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  std::optional<SymbolRefAttr> baseRef = getBaseRef();
  if (!baseRef)
    return success();

  return checkSymbolIsTypeOrAttribute(symbolTable, *this, *baseRef);
}

LogicalResult
ParametricOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  std::optional<SymbolRefAttr> baseRef = getBaseType();
  if (!baseRef)
    return success();

  return checkSymbolIsTypeOrAttribute(symbolTable, *this, *baseRef);
}

/// Parse a value with its variadicity first. By default, the variadicity is
/// single.
///
/// value-with-variadicity ::= ("single" | "optional" | "variadic")? ssa-value
static ParseResult
parseValueWithVariadicity(OpAsmParser &p,
                          OpAsmParser::UnresolvedOperand &operand,
                          VariadicityAttr &variadicityAttr) {
  MLIRContext *ctx = p.getBuilder().getContext();

  // Parse the variadicity, if present
  if (p.parseOptionalKeyword("single").succeeded()) {
    variadicityAttr = VariadicityAttr::get(ctx, Variadicity::single);
  } else if (p.parseOptionalKeyword("optional").succeeded()) {
    variadicityAttr = VariadicityAttr::get(ctx, Variadicity::optional);
  } else if (p.parseOptionalKeyword("variadic").succeeded()) {
    variadicityAttr = VariadicityAttr::get(ctx, Variadicity::variadic);
  } else {
    variadicityAttr = VariadicityAttr::get(ctx, Variadicity::single);
  }

  // Parse the value
  if (p.parseOperand(operand))
    return failure();
  return success();
}

static ParseResult parseNamedValuesWithVariadicity(
    OpAsmParser &p,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &attrOperands,
    ArrayAttr &attrNamesAttr, VariadicityArrayAttr &variadicityAttr) {
  Builder &builder = p.getBuilder();
  MLIRContext *ctx = builder.getContext();
  SmallVector<Attribute> attrNames;
  SmallVector<VariadicityAttr> variadicities;

  if (succeeded(p.parseOptionalLBrace())) {
    auto parseOperands = [&]() {
      if (p.parseAttribute(attrNames.emplace_back()) || p.parseEqual() ||
          parseValueWithVariadicity(p, attrOperands.emplace_back(),
                                    variadicities.emplace_back()))
        return failure();
      return success();
    };
    if (p.parseCommaSeparatedList(parseOperands) || p.parseRBrace())
      return failure();
  }
  attrNamesAttr = builder.getArrayAttr(attrNames);
  variadicityAttr = VariadicityArrayAttr::get(ctx, variadicities);
  return success();
}

static void
printNamedValuesWithVariadicity(OpAsmPrinter &p, Operation *op,
                                OperandRange attrArgs, ArrayAttr attrNames,
                                VariadicityArrayAttr variadicityAttr) {
  if (attrNames.empty())
    return;
  p << "{";
  p.increaseIndent();
  p.printNewline();
  interleave(
      llvm::seq<int>(0, attrNames.size()),
      [&](int i) {
        Variadicity variadicity = variadicityAttr[i].getValue();
        p << attrNames[i] << " = ";
        if (variadicity != Variadicity::single)
          p << stringifyVariadicity(variadicity) << " ";
        p << attrArgs[i];
      },
      [&] {
        p << ",";
        p.printNewline();
      });
  p.decreaseIndent();
  p.printNewline();
  p << '}';
}

LogicalResult RegionOp::verify() {
  if (IntegerAttr numberOfBlocks = getNumberOfBlocksAttr())
    if (int64_t number = numberOfBlocks.getInt(); number <= 0) {
      return emitOpError("the number of blocks is expected to be >= 1 but got ")
             << number;
    }
  return success();
}

#include "mlir/Dialect/IRDL/IR/IRDLInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLTypesGen.cpp.inc"

#include "mlir/Dialect/IRDL/IR/IRDLEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLOps.cpp.inc"
