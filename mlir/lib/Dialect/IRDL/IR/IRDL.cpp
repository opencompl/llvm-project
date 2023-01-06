//===- IRDL.cpp - IRDL dialect ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.h"
#include "mlir/Dialect/IRDL/TypeWrapper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::irdl;

using ArgDef = std::pair<StringRef, Attribute>;
using ArgDefs = ArrayRef<ArgDef>;

//===----------------------------------------------------------------------===//
// IRDL dialect.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDLDialect.cpp.inc"

void IRDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/IRDL/IR/IRDLOps.cpp.inc"
      >();
  registerAttributes();
}

void IRDLDialect::addTypeWrapper(std::unique_ptr<TypeWrapper> wrapper) {
  this->irdlContext.addTypeWrapper(std::move(wrapper));
}

TypeWrapper *IRDLDialect::getTypeWrapper(StringRef typeName) {
  return this->irdlContext.getTypeWrapper(typeName);
}

//===----------------------------------------------------------------------===//
// Operation parsing/printing
//===----------------------------------------------------------------------===//

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

/// Parse a dialect, an operation, or a type definition.
/// It parses the following syntax:
/// <symbol> <named-attr-dict> <region>
ParseResult parseIrdlDefinition(OpAsmParser &parser, OperationState &op) {
  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             op.attributes))
    return failure();

  // If extra attributes are present, parse them.
  NamedAttrList parsedAttributes;
  SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  auto disallowed = SymbolTable::getSymbolAttrName();
  if (parsedAttributes.get(disallowed))
    return parser.emitError(attributeDictLocation, "'")
           << disallowed
           << "' is an inferred attribute and should not be specified in the "
              "explicit attribute dictionary";
  op.attributes.append(parsedAttributes);

  auto *body = op.addRegion();
  if (parseSingleBlockRegion(parser, *body))
    return failure();

  return success();
}

void printIrdlDefinition(OpAsmPrinter &p, StringRef name,
                         DictionaryAttr::ValueType attrDict, Region &body,
                         Operation *op) {
  p << " ";
  p.printSymbolName(name);
  p << " ";
  if (!attrDict.empty()) {
    p.printOptionalAttrDict(attrDict, {SymbolTable::getSymbolAttrName()});
    p << " ";
  }
  printSingleBlockRegion(p, op, body);
}

ParseResult DialectOp::parse(OpAsmParser &parser, OperationState &op) {
  return parseIrdlDefinition(parser, op);
}

void DialectOp::print(OpAsmPrinter &p) {
  printIrdlDefinition(p, getName(), (*this)->getAttrDictionary().getValue(),
                      getBody(), getOperation());
}

ParseResult TypeOp::parse(OpAsmParser &parser, OperationState &op) {
  return parseIrdlDefinition(parser, op);
}

void TypeOp::print(OpAsmPrinter &p) {
  printIrdlDefinition(p, getName(), (*this)->getAttrDictionary().getValue(),
                      getBody(), getOperation());
}

ParseResult OperationOp::parse(OpAsmParser &parser, OperationState &op) {
  return parseIrdlDefinition(parser, op);
}

void OperationOp::print(OpAsmPrinter &p) {
  printIrdlDefinition(p, getName(), (*this)->getAttrDictionary().getValue(),
                      getBody(), getOperation());
}

//===----------------------------------------------------------------------===//
// Type definition reference.
//===----------------------------------------------------------------------===//

namespace {
OptionalParseResult parseOptionalTypeDefRef(OpAsmParser &p,
                                            TypeDefRefAttr *attrRes) {
  auto loc = p.getCurrentLocation();

  // Symref case
  {
    Attribute attr;
    auto res = p.parseOptionalAttribute(attr);

    if (res.has_value()) {
      if (res.value().failed())
        return res.value();
      if (auto symRef = dyn_cast<SymbolRefAttr>(attr))
        *attrRes = TypeDefRefAttr::get(symRef);
      return {success()};
    }
  }

  // Type wrapper case
  StringRef name;
  auto res = p.parseOptionalKeyword(&name);
  if (res.failed())
    return {};

  auto ctx = p.getBuilder().getContext();
  auto irdl = ctx->getOrLoadDialect<IRDLDialect>();
  auto *typeWrapper = irdl->getTypeWrapper(name);
  if (!typeWrapper) {
    p.emitError(loc, name + " is not a registered type wrapper");
    return {failure()};
  }
  *attrRes = TypeDefRefAttr::get(ctx, typeWrapper);
  return {success()};
}

void printTypeDefRef(OpAsmPrinter &p, TypeDefRefAttr attr) {
  if (auto symRef = attr.getSymRef())
    p.printAttribute(symRef);
  else
    p << attr.getTypeWrapper()->getName();
}
} // namespace

//===----------------------------------------------------------------------===//
// Type constraints.
//===----------------------------------------------------------------------===//

namespace {
ParseResult parseTypeConstraint(OpAsmParser &p, Attribute *typeConstraint);
void printTypeConstraint(OpAsmPrinter &p, Attribute typeConstraint);

/// Parse an Any constraint if there is one.
/// It has the format 'Any'
OptionalParseResult parseOptionalAnyTypeConstraint(OpAsmParser &p,
                                                   Attribute *typeConstraint) {
  if (p.parseOptionalKeyword("Any"))
    return {};

  *typeConstraint = AnyTypeConstraintAttr::get(p.getBuilder().getContext());
  return {success()};
}

/// Parse an AnyOf constraint if there is one.
/// It has the format 'AnyOf<type (, type)*>'
OptionalParseResult
parseOptionalAnyOfTypeConstraint(OpAsmParser &p, Attribute *typeConstraint) {
  if (p.parseOptionalKeyword("AnyOf"))
    return {};

  if (p.parseLess())
    return {failure()};

  SmallVector<Attribute> constraints;

  {
    Attribute constraint;
    if (parseTypeConstraint(p, &constraint))
      return {failure()};
    constraints.push_back(constraint);
  }

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return {failure()};

    Attribute constraint;
    if (parseTypeConstraint(p, &constraint))
      return {failure()};
    constraints.push_back(constraint);
  }

  *typeConstraint =
      AnyOfTypeConstraintAttr::get(p.getBuilder().getContext(), constraints);
  return {success()};
}

/// Print an AnyOf type constraint.
/// It has the format 'AnyOf<type, (, type)*>'.
void printAnyOfTypeConstraint(OpAsmPrinter &p,
                              AnyOfTypeConstraintAttr anyOfConstr) {
  auto constrs = anyOfConstr.getConstrs();

  p << "AnyOf<";
  for (size_t i = 0; i + 1 < constrs.size(); i++) {
    printTypeConstraint(p, constrs[i]);
    p << ", ";
  }
  printTypeConstraint(p, constrs.back());
  p << ">";
}

/// Parse an And constraint if there is one.
/// It has the format 'And<type (, type)*>'.
OptionalParseResult parseOptionalAndTypeConstraint(OpAsmParser &p,
                                                   Attribute *typeConstraint) {
  if (p.parseOptionalKeyword("And"))
    return {};

  if (p.parseLess())
    return {failure()};

  SmallVector<Attribute> constraints;

  {
    Attribute constraint;
    if (parseTypeConstraint(p, &constraint))
      return {failure()};
    constraints.push_back(constraint);
  }

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return {failure()};

    Attribute constraint;
    if (parseTypeConstraint(p, &constraint))
      return {failure()};
    constraints.push_back(constraint);
  }

  *typeConstraint =
      AndTypeConstraintAttr::get(p.getBuilder().getContext(), constraints);
  return {success()};
}

/// Print an And type constraint.
/// It has the format 'And<type (, type)*>'.
void printAndTypeConstraint(OpAsmPrinter &p, AndTypeConstraintAttr andConstr) {
  auto constrs = andConstr.getConstrs();

  p << "And<";
  for (size_t i = 0; i + 1 < constrs.size(); i++) {
    printTypeConstraint(p, constrs[i]);
    p << ", ";
  }
  printTypeConstraint(p, constrs.back());
  p << ">";
}

/// Parse a type parameters constraint.
/// It has the format 'dialectname.typename<(typeConstraint ,)*>'
ParseResult parseTypeParamsConstraint(OpAsmParser &p, TypeDefRefAttr typeDef,
                                      Attribute *typeConstraint) {
  auto ctx = p.getBuilder().getContext();

  // Empty case
  if (p.parseOptionalGreater().succeeded()) {
    *typeConstraint = TypeParamsConstraintAttr::get(ctx, typeDef, {});
    return success();
  }

  SmallVector<Attribute> paramConstraints;

  paramConstraints.push_back({});
  if (parseTypeConstraint(p, &paramConstraints.back()))
    return failure();

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return failure();

    paramConstraints.push_back({});
    if (parseTypeConstraint(p, &paramConstraints.back()))
      return failure();
  }

  *typeConstraint =
      TypeParamsConstraintAttr::get(ctx, typeDef, paramConstraints);
  return success();
}

void printTypeParamsConstraint(OpAsmPrinter &p,
                               TypeParamsConstraintAttr constraint) {
  printTypeDefRef(p, constraint.getTypeDef());

  auto paramConstraints = constraint.getParamConstraints();

  p << "<";
  llvm::interleaveComma(paramConstraints, p,
                        [&p](Attribute a) { printTypeConstraint(p, a); });
  p << ">";
}

/// Parse a type constraint.
/// The verifier ensures that the format is respected.
ParseResult parseTypeConstraint(OpAsmParser &p, Attribute *typeConstraint) {
  auto loc = p.getCurrentLocation();

  // Parse an Any constraint.
  auto anyRes = parseOptionalAnyTypeConstraint(p, typeConstraint);
  if (anyRes.has_value())
    return *anyRes;

  // Parse an AnyOf constraint.
  auto anyOfRes = parseOptionalAnyOfTypeConstraint(p, typeConstraint);
  if (anyOfRes.has_value())
    return *anyOfRes;

  // Parse an And constraint.
  auto andRes = parseOptionalAndTypeConstraint(p, typeConstraint);
  if (andRes.has_value())
    return *andRes;

  auto ctx = p.getBuilder().getContext();

  // Type equality constraint.
  // It has the format 'type'.
  Type type;
  auto typeParsed = p.parseOptionalType(type);
  if (typeParsed.has_value()) {
    if (failed(typeParsed.value()))
      return failure();

    *typeConstraint = EqTypeConstraintAttr::get(ctx, type);
    return success();
  }

  if (succeeded(p.parseOptionalQuestion())) {
    StringRef keyword;
    if (failed(p.parseKeyword(&keyword)))
      return failure();
    *typeConstraint = VarTypeConstraintAttr::get(ctx, keyword);
    return success();
  }

  TypeDefRefAttr typeDef;
  auto typeDefParsed = parseOptionalTypeDefRef(p, &typeDef);
  if (typeDefParsed.has_value()) {
    if (typeDefParsed.value().failed())
      return typeDefParsed.value();

    // Parameter constraints case
    if (p.parseOptionalLess().succeeded())
      return parseTypeParamsConstraint(p, typeDef, typeConstraint);

    // Base type constraint case
    *typeConstraint = TypeBaseConstraintAttr::get(ctx, typeDef);
    return success();
  }

  p.emitError(loc, "type constraint expected");
  return failure();
}

/// Print a type constraint.
void printTypeConstraint(OpAsmPrinter &p, Attribute typeConstraint) {
  if (auto eqConstr = typeConstraint.dyn_cast<EqTypeConstraintAttr>()) {
    p << eqConstr.getType();
  } else if (auto anyConstr =
                 typeConstraint.dyn_cast<AnyTypeConstraintAttr>()) {
    p << "Any";
  } else if (auto anyOfConstr =
                 typeConstraint.dyn_cast<AnyOfTypeConstraintAttr>()) {
    printAnyOfTypeConstraint(p, anyOfConstr);
  } else if (auto andConstr =
                 typeConstraint.dyn_cast<AndTypeConstraintAttr>()) {
    printAndTypeConstraint(p, andConstr);
  } else if (auto typeParamsConstr =
                 typeConstraint.dyn_cast<TypeParamsConstraintAttr>()) {
    printTypeParamsConstraint(p, typeParamsConstr);
  } else if (auto typeBaseConstr =
                 typeConstraint.dyn_cast<TypeBaseConstraintAttr>()) {
    auto typeDef = typeBaseConstr.getTypeDef();
    printTypeDefRef(p, typeDef);
  } else if (auto typeConstraintParam =
                 typeConstraint.dyn_cast<VarTypeConstraintAttr>()) {
    p << "?" << typeConstraintParam.getName();
  } else {
    assert(false && "Unknown type constraint.");
  }
}
} // namespace

//===----------------------------------------------------------------------===//
// irdl::DialectOp
//===----------------------------------------------------------------------===//

LogicalResult DialectOp::verify() {
  return success(Dialect::isValidNamespace(getName()));
}

//===----------------------------------------------------------------------===//
// NamedTypeConstraintArray
//===----------------------------------------------------------------------===//

ParseResult parseNamedTypeConstraint(OpAsmParser &p,
                                     NamedTypeConstraintAttr &param) {
  std::string name;
  if (failed(p.parseKeywordOrString(&name)))
    return failure();
  if (failed(p.parseColon()))
    return failure();
  Attribute attr;
  if (failed(parseTypeConstraint(p, &attr)))
    return failure();
  param = NamedTypeConstraintAttr::get(p.getContext(), name, attr);
  return success();
}

void printNamedTypeConstraint(OpAsmPrinter &p, NamedTypeConstraintAttr attr) {
  p.printKeywordOrString(attr.getName());
  p << ": ";
  printTypeConstraint(p, attr.getConstraint());
}

ParseResult parseNamedTypeConstraintArray(OpAsmParser &p,
                                          ArrayAttr &paramsAttr) {
  SmallVector<Attribute> attrs;
  auto parseRes = p.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        NamedTypeConstraintAttr attr;
        if (failed(parseNamedTypeConstraint(p, attr)))
          return failure();
        attrs.push_back(attr);
        return success();
      });
  if (parseRes.failed())
    return failure();
  paramsAttr = ArrayAttr::get(p.getContext(), attrs);
  return success();
}

void printNamedTypeConstraintArray(OpAsmPrinter &p, Operation *,
                                   ArrayAttr paramsAttr) {
  p << "(";
  llvm::interleaveComma(paramsAttr.getValue(), p, [&](Attribute attr) {
    printNamedTypeConstraint(p, attr.cast<NamedTypeConstraintAttr>());
  });
  p << ")";
}

//===----------------------------------------------------------------------===//
// IRDL operations.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLOps.cpp.inc"

//===----------------------------------------------------------------------===//
// IRDL interfaces.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDLInterfaces.cpp.inc"
