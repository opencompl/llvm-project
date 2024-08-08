//===- OpDefinitionsGen.cpp - IRDL op definitions generator ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpDefinitionsGen uses the description of operations to generate IRDL
// definitions for ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;
using tblgen::NamedTypeConstraint;

static llvm::cl::OptionCategory dialectGenCat("Options for -gen-irdl-dialect");
llvm::cl::opt<std::string>
    selectedDialect("dialect", llvm::cl::desc("The dialect to gen for"),
                    llvm::cl::cat(dialectGenCat), llvm::cl::Required);

Value typeToConstraint(OpBuilder &builder, MLIRContext *ctx, Type type) {
  auto op =
      builder.create<irdl::IsOp>(UnknownLoc::get(ctx), TypeAttr::get(type));
  return op.getOutput();
}
Value fromWidthSignedness(OpBuilder &builder, MLIRContext *ctx, int64_t width,
                          IntegerType::SignednessSemantics signedness) {
  return typeToConstraint(builder, ctx,
                          IntegerType::get(ctx, width, signedness));
}

Value createIntConstraint(
    OpBuilder &builder, const Record &predRec,
    std::optional<IntegerType::SignednessSemantics> signedness = {}) {
  MLIRContext *ctx = builder.getContext();
  auto width = predRec.getValueAsInt("bitwidth");
  if (signedness.has_value()) {
    return fromWidthSignedness(builder, ctx, width, signedness.value());
  }
  std::vector<Value> types = {
      fromWidthSignedness(builder, ctx, width, IntegerType::Signless),
      fromWidthSignedness(builder, ctx, width, IntegerType::Signed),
      fromWidthSignedness(builder, ctx, width, IntegerType::Unsigned)};
  auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), types);
  return op.getOutput();
}

Value createConstraint(OpBuilder &builder, tblgen::Constraint constraint) {
  MLIRContext *ctx = builder.getContext();
  const Record &predRec = constraint.getDef();

  if (predRec.isSubClassOf("Variadic") || predRec.isSubClassOf("Optional")) {
    return createConstraint(builder, predRec.getValueAsDef("baseType"));
  }

  if (predRec.getName() == "AnyType") {
    auto op = builder.create<irdl::AnyOp>(UnknownLoc::get(ctx));
    return op.getOutput();
  }

  if (predRec.getName() == "NoneType") {
    return typeToConstraint(builder, ctx, NoneType::get(ctx));
  }

  if (predRec.isSubClassOf("TypeDef")) {
    std::string typeName = ("!" + predRec.getValueAsString("typeName")).str();
    auto op = builder.create<irdl::BaseOp>(UnknownLoc::get(ctx),
                                           StringAttr::get(ctx, typeName));
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AnyTypeOf")) {
    std::vector<Value> constraints;
    for (Record *child : predRec.getValueAsListOfDefs("allowedTypes")) {
      constraints.push_back(
          createConstraint(builder, tblgen::Constraint(child)));
    }
    auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AllOfType")) {
    std::vector<Value> constraints;
    for (Record *child : predRec.getValueAsListOfDefs("allowedTypes")) {
      constraints.push_back(
          createConstraint(builder, tblgen::Constraint(child)));
    }
    auto op = builder.create<irdl::AllOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  // Integer types
  if (predRec.isSubClassOf("AnyInteger")) {
    auto op = builder.create<irdl::BaseOp>(
        UnknownLoc::get(ctx), StringAttr::get(ctx, "!builtin.integer"));
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AnyI")) {
    return createIntConstraint(builder, predRec);
  }

  if (predRec.isSubClassOf("I")) {
    return createIntConstraint(builder, predRec,
                               IntegerType::SignednessSemantics::Signless);
  }

  if (predRec.isSubClassOf("SI")) {
    return createIntConstraint(builder, predRec,
                               IntegerType::SignednessSemantics::Signed);
  }

  if (predRec.isSubClassOf("UI")) {
    return createIntConstraint(builder, predRec,
                               IntegerType::SignednessSemantics::Unsigned);
  }

  // Index type
  if (predRec.getName() == "Index") {
    return typeToConstraint(builder, ctx, IndexType::get(ctx));
  }

  // Float types
  if (predRec.isSubClassOf("F")) {
    auto width = predRec.getValueAsInt("bitwidth");
    Type type;
    bool typeFound = true;
    switch (width) {
    case 16:
      type = FloatType::getF16(ctx);
      break;
    case 32:
      type = FloatType::getF32(ctx);
      break;
    case 64:
      type = FloatType::getF64(ctx);
      break;
    case 80:
      type = FloatType::getF80(ctx);
      break;
    case 128:
      type = FloatType::getF128(ctx);
      break;
    default:
      typeFound = false;
      break;
    }
    if (typeFound) {
      return typeToConstraint(builder, ctx, type);
    }
  }

  if (predRec.getName() == "BF16") {
    return typeToConstraint(builder, ctx, FloatType::getBF16(ctx));
  }

  if (predRec.getName() == "TF32") {
    return typeToConstraint(builder, ctx, FloatType::getTF32(ctx));
  }

  if (predRec.getName() == "F8E4M3FN") {
    return typeToConstraint(builder, ctx, FloatType::getFloat8E4M3(ctx));
  }

  if (predRec.getName() == "F8E5M2") {
    return typeToConstraint(builder, ctx, FloatType::getFloat8E5M2(ctx));
  }

  if (predRec.getName() == "F8E4M3") {
    return typeToConstraint(builder, ctx, FloatType::getFloat8E4M3(ctx));
  }

  if (predRec.getName() == "F8E4M3FNUZ") {
    return typeToConstraint(builder, ctx, FloatType::getFloat8E4M3FNUZ(ctx));
  }

  if (predRec.getName() == "F8E4M3B11FNUZ") {
    return typeToConstraint(builder, ctx, FloatType::getFloat8E4M3B11FNUZ(ctx));
  }

  if (predRec.getName() == "F8E5M2FNUZ") {
    return typeToConstraint(builder, ctx, FloatType::getFloat8E5M2FNUZ(ctx));
  }

  if (predRec.getName() == "F8E3M4") {
    return typeToConstraint(builder, ctx, FloatType::getFloat8E3M4(ctx));
  }

  std::string condition = constraint.getPredicate().getCondition();
  // Build a CPredOp to match the C constraint built.
  irdl::CPredOp op = builder.create<irdl::CPredOp>(
      UnknownLoc::get(ctx), StringAttr::get(ctx, condition));
  return op;
}

/// Returns the name of the operation without the dialect prefix.
static StringRef getOperatorName(tblgen::Operator &tblgenOp) {
  StringRef opName = tblgenOp.getDef().getValueAsString("opName");
  return opName;
}

/// Extract an operation to IRDL.
irdl::OperationOp createIRDLOperation(OpBuilder &builder,
                                      tblgen::Operator &tblgenOp) {
  MLIRContext *ctx = builder.getContext();
  StringRef opName = getOperatorName(tblgenOp);

  irdl::OperationOp op = builder.create<irdl::OperationOp>(
      UnknownLoc::get(ctx), StringAttr::get(ctx, opName));

  // Add the block in the region.
  Block &opBlock = op.getBody().emplaceBlock();
  OpBuilder consBuilder = OpBuilder::atBlockBegin(&opBlock);

  auto getValues = [&](tblgen::Operator::const_value_range namedCons) {
    SmallVector<Value> operands;
    SmallVector<irdl::VariadicityAttr> variadicity;
    for (const NamedTypeConstraint &namedCons : namedCons) {
      auto operand = createConstraint(consBuilder, namedCons.constraint);
      operands.push_back(operand);

      irdl::VariadicityAttr var;
      if (namedCons.isOptional())
        var = consBuilder.getAttr<irdl::VariadicityAttr>(
            irdl::Variadicity::optional);
      else if (namedCons.isVariadic())
        var = consBuilder.getAttr<irdl::VariadicityAttr>(
            irdl::Variadicity::variadic);
      else
        var = consBuilder.getAttr<irdl::VariadicityAttr>(
            irdl::Variadicity::single);

      variadicity.push_back(var);
    }
    return std::make_tuple(operands, variadicity);
  };

  auto [operands, operandVariadicity] = getValues(tblgenOp.getOperands());
  auto [results, resultVariadicity] = getValues(tblgenOp.getResults());

  // Create the operands and results operations.
  consBuilder.create<irdl::OperandsOp>(UnknownLoc::get(ctx), operands,
                                       operandVariadicity);
  consBuilder.create<irdl::ResultsOp>(UnknownLoc::get(ctx), results,
                                      resultVariadicity);

  return op;
}

static irdl::DialectOp createIRDLDialect(OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  return builder.create<irdl::DialectOp>(UnknownLoc::get(ctx),
                                         StringAttr::get(ctx, selectedDialect));
}

static std::vector<llvm::Record *>
getOpDefinitions(const RecordKeeper &recordKeeper) {
  if (!recordKeeper.getClass("Op"))
    return {};
  return recordKeeper.getAllDerivedDefinitions("Op");
}

static bool emitDialectIRDLDefs(const RecordKeeper &recordKeeper,
                                raw_ostream &os) {
  // Initialize.
  MLIRContext ctx;
  ctx.getOrLoadDialect<irdl::IRDLDialect>();
  OpBuilder builder(&ctx);

  // Create a module op and set it as the insertion point.
  OwningOpRef<ModuleOp> module =
      builder.create<ModuleOp>(UnknownLoc::get(&ctx));
  builder = builder.atBlockBegin(module->getBody());
  // Create the dialect and insert it.
  irdl::DialectOp dialect = createIRDLDialect(builder);
  // Set insertion point to start of DialectOp.
  builder = builder.atBlockBegin(&dialect.getBody().emplaceBlock());

  std::vector<Record *> defs = getOpDefinitions(recordKeeper);
  for (auto *def : defs) {
    tblgen::Operator tblgenOp(def);
    if (tblgenOp.getDialectName() != selectedDialect)
      continue;

    createIRDLOperation(builder, tblgenOp);
  }

  // Print the module.
  module->print(os);

  return false;
}

static mlir::GenRegistration
    genOpDefs("gen-dialect-irdl-defs", "Generate IRDL dialect definitions",
              [](const RecordKeeper &records, raw_ostream &os) {
                return emitDialectIRDLDefs(records, os);
              });
