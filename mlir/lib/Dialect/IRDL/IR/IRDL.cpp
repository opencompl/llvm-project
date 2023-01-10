//===- IRDL.cpp - IRDL dialect ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.h"
#include "mlir/Dialect/IRDL/IRDLRegistration.h"
#include "mlir/Dialect/IRDL/TypeWrapper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::irdl;
using mlir::irdl::TypeWrapper;

using ArgDef = std::pair<StringRef, Attribute>;
using ArgDefs = ArrayRef<ArgDef>;

//===----------------------------------------------------------------------===//
// IRDL dialect.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.cpp.inc"

#include "mlir/Dialect/IRDL/IR/IRDLDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"

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

void IRDLDialect::addTypeWrapper(std::unique_ptr<TypeWrapper> wrapper) {
  this->irdlContext.addTypeWrapper(std::move(wrapper));
}

TypeWrapper *IRDLDialect::getTypeWrapper(StringRef typeName) {
  return this->irdlContext.getTypeWrapper(typeName);
}

//===----------------------------------------------------------------------===//
// Parsing/Printing
//===----------------------------------------------------------------------===//

static ParseResult parseKeywordOrString(OpAsmParser &p, StringAttr &attr) {
  std::string str;
  if (failed(p.parseKeywordOrString(&str)))
    return failure();
  attr = p.getBuilder().getStringAttr(str);
  return success();
}

static void printKeywordOrString(OpAsmPrinter &p, Operation *,
                                 StringAttr attr) {
  p.printKeywordOrString(attr.getValue());
}

/// Parse a region, and add a single block if the region is empty.
/// If no region is parsed, create a new region with a single empty block.
static ParseResult parseSingleBlockRegion(OpAsmParser &p, Region &region) {
  auto regionParseRes = p.parseOptionalRegion(region);
  if (regionParseRes.has_value()) {
    if (failed(regionParseRes.value()))
      return failure();
  }
  // If the region is empty, add a single empty block.
  if (region.getBlocks().size() == 0) {
    region.push_back(new Block());
  }

  return success();
}

static void printSingleBlockRegion(OpAsmPrinter &p, Operation *op,
                                   Region &region) {
  if (!region.getBlocks().front().empty()) {
    p.printRegion(region);
  }
}

LogicalResult DialectOp::verify() {
  return success(Dialect::isValidNamespace(getName()));
}

#include "mlir/Dialect/IRDL/IR/IRDLInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLTypesGen.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLOps.cpp.inc"
