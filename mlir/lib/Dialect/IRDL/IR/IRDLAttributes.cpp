//===- IRDLAttributes.cpp - IRDL dialect ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/AttributeWrapper.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"

using namespace mlir;
using namespace mlir::irdl;

LogicalResult TypeOrAttrDefRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, TypeWrapper *typeWrapper,
    AttributeWrapper *attrWrapper, SymbolRefAttr symRef) {
  if (!typeWrapper && !attrWrapper && !symRef) {
    emitError()
        << "expected a type wrapper, attribute wrapper, or symbol reference";
    return failure();
  }
  if ((symRef && (attrWrapper || typeWrapper)) ||
      (attrWrapper && typeWrapper)) {
    emitError() << "expected either a type wrapper, an attribute wrapper, or a "
                   "symbol reference, but not more than one";
    return failure();
  }
  return success();
}

Attribute TypeOrAttrDefRefAttr::parse(AsmParser &parser, Type type) {
  auto loc = parser.getCurrentLocation();

  std::string stringLit;
  auto litStringRes = parser.parseOptionalString(&stringLit);

  // Wrapper cases.
  if (litStringRes.succeeded()) {
    auto ctx = parser.getBuilder().getContext();
    auto irdl = ctx->getOrLoadDialect<IRDLDialect>();

    if (stringLit.empty() || (stringLit[0] != '!' && stringLit[0] != '#')) {
      parser.emitError(loc) << "attribute or type wrapper name should "
                               "be prefixed with either '!' or '#'";
      return {};
    }

    StringRef wrapperName = StringRef(stringLit).substr(1);

    // Type wrapper case
    if (stringLit[0] == '!') {
      auto *typeWrapper = irdl->irdlContext.getTypeWrapper(wrapperName);
      if (typeWrapper)
        return TypeOrAttrDefRefAttr::get(ctx, typeWrapper);
      parser.emitError(loc)
          << "type wrapper " << wrapperName << " is not registered";
      return {};
    }

    // Attribute wrapper case
    auto *attrWrapper = irdl->irdlContext.getAttributeWrapper(wrapperName);
    if (attrWrapper)
      return TypeOrAttrDefRefAttr::get(ctx, attrWrapper);

    parser.emitError(loc) << "attribute wrapper " << wrapperName
                          << " is not registered";
    return {};
  }

  // Symref cases
  Attribute attr;
  auto res = parser.parseAttribute(attr);

  if (res.failed())
    return {};
  if (auto symRef = llvm::dyn_cast<SymbolRefAttr>(attr))
    return TypeOrAttrDefRefAttr::get(symRef);
  parser.emitError(loc) << "expected a string literal, or a symbol reference";
  return {};
}

void TypeOrAttrDefRefAttr::print(AsmPrinter &odsPrinter) const {
  if (auto *typeWrapper = getTypeWrapper())
    odsPrinter << "\"!" << typeWrapper->getName() << '"';
  else if (auto *attrWrapper = getAttrWrapper())
    odsPrinter << "\"#" << attrWrapper->getName() << '"';
  else if (auto symRef = getSymRef())
    odsPrinter << symRef;
  else
    llvm_unreachable("expected a type wrapper, attribute wrapper, or symbol "
                     "reference");
}
