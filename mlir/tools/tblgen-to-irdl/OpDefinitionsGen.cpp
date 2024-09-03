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

Value typeToConstraint(OpBuilder &builder, Type type) {
  MLIRContext *ctx = builder.getContext();
  auto op =
      builder.create<irdl::IsOp>(UnknownLoc::get(ctx), TypeAttr::get(type));
  return op.getOutput();
}

Value baseToConstraint(OpBuilder &builder, StringRef baseClass) {
  MLIRContext *ctx = builder.getContext();
  auto op = builder.create<irdl::BaseOp>(UnknownLoc::get(ctx),
                                         StringAttr::get(ctx, baseClass));
  return op.getOutput();
}

Value getAllFloats(OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  SmallVector<Value> values;
  std::string builtin = "!builtin.";
  for (const char *x : {"f8E5M2", "f8E4M3", "f8E4M3FN", "f8E5M2FNUZ",
                        "f8E4M3FNUZ", "f8E4M3B11FNUZ", "f8E3M4", "bf16", "f16",
                        "tf32", "f32", "f64", "f80", "f128"}) {
    values.push_back(baseToConstraint(builder, builtin + x));
  }

  auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), values);
  return op.getOutput();
}

Value createPredicate(OpBuilder &builder, tblgen::Pred pred) {
  MLIRContext *ctx = builder.getContext();

  const Record &predRec = pred.getDef();

  if (predRec.isSubClassOf("HasAnyRankOfPred")) {
    auto ranks = predRec.getValueAsListOfInts("ranks");
    std::vector<Value> constraints;

    for (auto rank : ranks) {
      auto ty = IntegerType::get(ctx, 32);
      auto op = builder.create<irdl::HasRankOp>(UnknownLoc::get(ctx),
                                                IntegerAttr::get(ty, rank));
      constraints.push_back(op.getOutput());
    }

    auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  if (pred.isCombined()) {
    auto combiner = pred.getDef().getValueAsDef("kind")->getName();
    if (combiner == "PredCombinerAnd" || combiner == "PredCombinerOr") {
      std::vector<Value> constraints;
      for (auto *child : pred.getDef().getValueAsListOfDefs("children")) {
        constraints.push_back(createPredicate(builder, tblgen::Pred(child)));
      }
      if (combiner == "PredCombinerAnd") {
        auto op =
            builder.create<irdl::AllOfOp>(UnknownLoc::get(ctx), constraints);
        return op.getOutput();
      }
      auto op =
          builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), constraints);
      return op.getOutput();
    }
  }

  if (predRec.getName() == "IsRankedTensorTypePred") {
    return baseToConstraint(builder, "!builtin.tensor");
  }

  if (predRec.getName() == "IsUnrankedTensorTypePred") {
    return baseToConstraint(builder, "!builtin.unranked_tensor");
  }

  if (predRec.getName() == "IsTensorTypePred") {
    SmallVector<Value> constraints = {
        baseToConstraint(builder, "!builtin.tensor"),
        baseToConstraint(builder, "!builtin.unranked_tensor")};
    auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  if (predRec.getName() == "IsMemRefTypePred") {
    return baseToConstraint(builder, "!builtin.memref");
  }

  if (predRec.getName() == "IsUnrankedMemRefTypePred") {
    return baseToConstraint(builder, "!builtin.unranked_memref");
  }

  if (predRec.getName() == "IsBaseMemRefTypePred") {
    SmallVector<Value> constraints = {
        baseToConstraint(builder, "!builtin.memref"),
        baseToConstraint(builder, "!builtin.unranked_memref")};
    auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  std::string condition = pred.getCondition();
  // Build a CPredOp to match the C constraint built.
  irdl::CPredOp op = builder.create<irdl::CPredOp>(
      UnknownLoc::get(ctx), StringAttr::get(ctx, condition));
  return op;
}

std::optional<Type> recordToType(MLIRContext *ctx, const Record &predRec) {
  if (predRec.isSubClassOf("I")) {
    auto width = predRec.getValueAsInt("bitwidth");
    return IntegerType::get(ctx, width, IntegerType::Signless);
  }

  if (predRec.isSubClassOf("SI")) {
    auto width = predRec.getValueAsInt("bitwidth");
    return IntegerType::get(ctx, width, IntegerType::Signed);
  }

  if (predRec.isSubClassOf("UI")) {
    auto width = predRec.getValueAsInt("bitwidth");
    return IntegerType::get(ctx, width, IntegerType::Unsigned);
  }

  // Index type
  if (predRec.getName() == "Index") {
    return IndexType::get(ctx);
  }

  // Float types
  if (predRec.isSubClassOf("F")) {
    auto width = predRec.getValueAsInt("bitwidth");
    switch (width) {
    case 16:
      return FloatType::getF16(ctx);
    case 32:
      return FloatType::getF32(ctx);
    case 64:
      return FloatType::getF64(ctx);
    case 80:
      return FloatType::getF80(ctx);
    case 128:
      return FloatType::getF128(ctx);
    }
  }

  if (predRec.getName() == "NoneType") {
    return NoneType::get(ctx);
  }

  if (predRec.getName() == "BF16") {
    return FloatType::getBF16(ctx);
  }

  if (predRec.getName() == "TF32") {
    return FloatType::getTF32(ctx);
  }

  if (predRec.getName() == "F8E4M3FN") {
    return FloatType::getFloat8E4M3FN(ctx);
  }

  if (predRec.getName() == "F8E5M2") {
    return FloatType::getFloat8E5M2(ctx);
  }

  if (predRec.getName() == "F8E4M3") {
    return FloatType::getFloat8E4M3(ctx);
  }

  if (predRec.getName() == "F8E4M3FNUZ") {
    return FloatType::getFloat8E4M3FNUZ(ctx);
  }

  if (predRec.getName() == "F8E4M3B11FNUZ") {
    return FloatType::getFloat8E4M3B11FNUZ(ctx);
  }

  if (predRec.getName() == "F8E5M2FNUZ") {
    return FloatType::getFloat8E5M2FNUZ(ctx);
  }

  if (predRec.getName() == "F8E3M4") {
    return FloatType::getFloat8E3M4(ctx);
  }

  if (predRec.isSubClassOf("Complex")) {
    const Record *elementRec = predRec.getValueAsDef("elementType");
    auto elementType = recordToType(ctx, *elementRec);
    if (elementType.has_value()) {
      return ComplexType::get(elementType.value());
    }
  }

  return std::nullopt;
}

Value createTypeConstraint(OpBuilder &builder, tblgen::Constraint constraint) {
  MLIRContext *ctx = builder.getContext();
  const Record &predRec = constraint.getDef();

  if (predRec.isSubClassOf("Variadic") || predRec.isSubClassOf("Optional"))
    return createTypeConstraint(builder, predRec.getValueAsDef("baseType"));

  if (predRec.getName() == "AnyType") {
    auto op = builder.create<irdl::AnyOp>(UnknownLoc::get(ctx));
    return op.getOutput();
  }

  if (predRec.isSubClassOf("TypeDef")) {
    auto dialect = predRec.getValueAsDef("dialect")->getValueAsString("name");
    if (dialect == selectedDialect) {
      std::string combined = ("!" + predRec.getValueAsString("mnemonic")).str();
      SmallVector<FlatSymbolRefAttr> nested = {
          SymbolRefAttr::get(ctx, combined)};
      auto typeSymbol = SymbolRefAttr::get(ctx, dialect, nested);
      auto op = builder.create<irdl::BaseOp>(UnknownLoc::get(ctx), typeSymbol);
      return op.getOutput();
    }
    std::string typeName = ("!" + predRec.getValueAsString("typeName")).str();
    auto op = builder.create<irdl::BaseOp>(UnknownLoc::get(ctx),
                                           StringAttr::get(ctx, typeName));
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AnyTypeOf")) {
    std::vector<Value> constraints;
    for (const Record *child : predRec.getValueAsListOfDefs("allowedTypes")) {
      constraints.push_back(
          createTypeConstraint(builder, tblgen::Constraint(child)));
    }
    auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AllOfType")) {
    std::vector<Value> constraints;
    for (const Record *child : predRec.getValueAsListOfDefs("allowedTypes")) {
      constraints.push_back(
          createTypeConstraint(builder, tblgen::Constraint(child)));
    }
    auto op = builder.create<irdl::AllOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  // Shaped types
  if (predRec.isSubClassOf("ShapedContainerType")) {
    std::vector<Value> constraints;
    for (auto *type : predRec.getValueAsListOfDefs("allowedTypeList")) {
      auto typeConstraint = createTypeConstraint(builder, type);
      auto op = builder.create<irdl::HasElementTypeOp>(UnknownLoc::get(ctx),
                                                       typeConstraint);
      constraints.push_back(op.getOutput());
    }
    constraints.push_back(
        createPredicate(builder, tblgen::Pred(predRec.getValueAsDef("pred"))));
    auto op = builder.create<irdl::AllOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  if (predRec.getName() == "AnyFloat") {
    return getAllFloats(builder);
  }

  if (predRec.getName() == "AnyInteger" ||
      predRec.getName() == "AnySignlessInteger" ||
      predRec.getName() == "AnySignedInteger" ||
      predRec.getName() == "AnyUnsignedInteger") {
    return baseToConstraint(builder, "!builtin.integer");
  }

  if (predRec.getName() == "AnySignlessIntegerOrIndex") {
    SmallVector<Value> constraints = {
        baseToConstraint(builder, "!builtin.integer"),
        typeToConstraint(builder, IndexType::get(ctx))};
    auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AnyI")) {
    auto width = predRec.getValueAsInt("bitwidth");
    std::vector<Value> types = {
        typeToConstraint(builder,
                         IntegerType::get(ctx, width, IntegerType::Signless)),
        typeToConstraint(builder,
                         IntegerType::get(ctx, width, IntegerType::Signed)),
        typeToConstraint(builder,
                         IntegerType::get(ctx, width, IntegerType::Unsigned))};
    auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), types);
    return op.getOutput();
  }

  auto type = recordToType(ctx, predRec);

  if (type.has_value()) {
    return typeToConstraint(builder, type.value());
  }

  // Confined type
  if (predRec.isSubClassOf("ConfinedType")) {
    std::vector<Value> constraints;
    constraints.push_back(createTypeConstraint(
        builder, tblgen::Constraint(predRec.getValueAsDef("baseType"))));
    for (const Record *child : predRec.getValueAsListOfDefs("predicateList")) {
      constraints.push_back(createPredicate(builder, tblgen::Pred(child)));
    }
    auto op = builder.create<irdl::AllOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  return createPredicate(builder, constraint.getPredicate());
}

Value createAttrConstraint(OpBuilder &builder, tblgen::Constraint constraint) {
  MLIRContext *ctx = builder.getContext();
  const Record &predRec = constraint.getDef();

  if (predRec.isSubClassOf("DefaultValuedAttr") ||
      predRec.isSubClassOf("DefaultValuedOptionalAttr") ||
      predRec.isSubClassOf("OptionalAttr")) {
    return createAttrConstraint(builder, predRec.getValueAsDef("baseAttr"));
  }

  if (predRec.isSubClassOf("ConfinedAttr")) {
    std::vector<Value> constraints;
    constraints.push_back(createAttrConstraint(
        builder, tblgen::Constraint(predRec.getValueAsDef("baseAttr"))));
    for (const Record *child :
         predRec.getValueAsListOfDefs("attrConstraints")) {
      constraints.push_back(createPredicate(
          builder, tblgen::Pred(child->getValueAsDef("predicate"))));
    }
    auto op = builder.create<irdl::AllOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AnyAttrOf")) {
    std::vector<Value> constraints;
    for (const Record *child :
         predRec.getValueAsListOfDefs("allowedAttributes")) {
      constraints.push_back(
          createAttrConstraint(builder, tblgen::Constraint(child)));
    }
    auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  if (predRec.getName() == "AnyAttr") {
    auto op = builder.create<irdl::AnyOp>(UnknownLoc::get(ctx));
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AnyIntegerAttrBase") ||
      predRec.getName() == "AnySignlessIntegerAttrBase" ||
      predRec.isSubClassOf("SignlessIntegerAttrBase") ||
      predRec.getName() == "AnySignedIntegerAttrBase" ||
      predRec.isSubClassOf("SignedIntegerAttrBase") ||
      predRec.getName() == "AnyUnsignedIntegerAttrBase" ||
      predRec.isSubClassOf("UnsignedIntegerAttrBase") ||
      predRec.getName() == "BoolAttr" || predRec.getName() == "IndexAttr") {
    return baseToConstraint(builder, "!builtin.integer");
  }

  if (predRec.isSubClassOf("FloatAttrBase") ||
      predRec.getName() == "AnyFloat") {
    return getAllFloats(builder);
  }

  if (predRec.isSubClassOf("StringBasedAttr")) {
    return baseToConstraint(builder, "!builtin.string");
  }

  if (predRec.getName() == "UnitAttr") {
    auto op =
        builder.create<irdl::IsOp>(UnknownLoc::get(ctx), UnitAttr::get(ctx));
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AttrDef")) {
    auto dialect = predRec.getValueAsDef("dialect")->getValueAsString("name");
    if (dialect == selectedDialect) {
      std::string combined = ("#" + predRec.getValueAsString("mnemonic")).str();
      SmallVector<FlatSymbolRefAttr> nested = {SymbolRefAttr::get(ctx, combined)

      };
      auto typeSymbol = SymbolRefAttr::get(ctx, dialect, nested);
      auto op = builder.create<irdl::BaseOp>(UnknownLoc::get(ctx), typeSymbol);
      return op.getOutput();
    }
    std::string typeName = ("#" + predRec.getValueAsString("attrName")).str();
    auto op = builder.create<irdl::BaseOp>(UnknownLoc::get(ctx),
                                           StringAttr::get(ctx, typeName));
    return op.getOutput();
  }

  return createPredicate(builder, constraint.getPredicate());
}

Value createRegionConstraint(OpBuilder &builder, tblgen::Region constraint) {
  MLIRContext *ctx = builder.getContext();
  const Record &predRec = constraint.getDef();

  if (predRec.getName() == "AnyRegion") {
    ValueRange entryBlockArgs = {};
    auto op =
        builder.create<irdl::RegionOp>(UnknownLoc::get(ctx), entryBlockArgs);
    return op.getResult();
  }

  if (predRec.isSubClassOf("SizedRegion")) {
    ValueRange entryBlockArgs = {};
    auto ty = IntegerType::get(ctx, 32);
    auto op = builder.create<irdl::RegionOp>(
        UnknownLoc::get(ctx), entryBlockArgs,
        IntegerAttr::get(ty, predRec.getValueAsInt("blocks")));
    return op.getResult();
  }

  return createPredicate(builder, constraint.getPredicate());
}

/// Returns the name of the operation without the dialect prefix.
static StringRef getOperatorName(tblgen::Operator &tblgenOp) {
  StringRef opName = tblgenOp.getDef().getValueAsString("opName");
  return opName;
}

/// Returns the name of the type without the dialect prefix.
static StringRef getTypeName(tblgen::TypeDef &tblgenType) {
  StringRef opName = tblgenType.getDef()->getValueAsString("mnemonic");
  return opName;
}

/// Returns the name of the attr without the dialect prefix.
static StringRef getAttrName(tblgen::AttrDef &tblgenType) {
  StringRef opName = tblgenType.getDef()->getValueAsString("mnemonic");
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
    SmallVector<Attribute> opNames;
    SmallVector<irdl::VariadicityAttr> variadicity;
    for (const NamedTypeConstraint &namedCons : namedCons) {
      auto operand = createTypeConstraint(consBuilder, namedCons.constraint);
      operands.push_back(operand);

      opNames.push_back(StringAttr::get(ctx, namedCons.name));

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
    return std::make_tuple(operands, opNames, variadicity);
  };

  auto [operands, operandNames, operandVariadicity] =
      getValues(tblgenOp.getOperands());
  auto [results, resultNames, resultVariadicity] =
      getValues(tblgenOp.getResults());

  SmallVector<Value> attributes;
  SmallVector<Attribute> attrNames;
  SmallVector<irdl::VariadicityAttr> attrVariadicity;
  for (auto namedAttr : tblgenOp.getAttributes()) {
    irdl::VariadicityAttr var;
    if (namedAttr.attr.isOptional())
      var = consBuilder.getAttr<irdl::VariadicityAttr>(
          irdl::Variadicity::optional);
    else
      var =
          consBuilder.getAttr<irdl::VariadicityAttr>(irdl::Variadicity::single);
    attributes.push_back(createAttrConstraint(consBuilder, namedAttr.attr));
    attrNames.push_back(StringAttr::get(ctx, namedAttr.name));
    attrVariadicity.push_back(var);
  }

  SmallVector<Value> regions;
  SmallVector<Attribute> regionNames;
  SmallVector<irdl::VariadicityAttr> regionVariadicity;
  for (auto namedRegion : tblgenOp.getRegions()) {
    regionNames.push_back(StringAttr::get(ctx, namedRegion.name));
    irdl::VariadicityAttr var;
    if (namedRegion.isVariadic())
      var = consBuilder.getAttr<irdl::VariadicityAttr>(
          irdl::Variadicity::variadic);
    else
      var =
          consBuilder.getAttr<irdl::VariadicityAttr>(irdl::Variadicity::single);
    regions.push_back(
        createRegionConstraint(consBuilder, namedRegion.constraint));
    regionVariadicity.push_back(var);
  }

  // Create the operands and results operations.
  if (!operands.empty())
    consBuilder.create<irdl::OperandsOp>(UnknownLoc::get(ctx), operands,
                                         ArrayAttr::get(ctx, operandNames),
                                         operandVariadicity);
  if (!results.empty())
    consBuilder.create<irdl::ResultsOp>(UnknownLoc::get(ctx), results,
                                        ArrayAttr::get(ctx, resultNames),
                                        resultVariadicity);
  if (!attributes.empty())
    consBuilder.create<irdl::AttributesOp>(UnknownLoc::get(ctx), attributes,
                                           ArrayAttr::get(ctx, attrNames),
                                           attrVariadicity);
  if (!regions.empty())
    consBuilder.create<irdl::RegionsOp>(UnknownLoc::get(ctx), regions,
                                        ArrayAttr::get(ctx, regionNames),
                                        regionVariadicity);
  if (tblgenOp.hasSummary())
    consBuilder.create<irdl::SummaryOp>(
        UnknownLoc::get(ctx),
        StringAttr::get(ctx, tblgenOp.getSummary().trim()));
  if (tblgenOp.hasAssemblyFormat())
    consBuilder.create<irdl::AssemblyFormatOp>(
        UnknownLoc::get(ctx),
        StringAttr::get(ctx, tblgenOp.getAssemblyFormat().trim()));

  return op;
}

irdl::TypeOp createIRDLType(OpBuilder &builder, tblgen::TypeDef &tblgenType) {
  MLIRContext *ctx = builder.getContext();
  StringRef typeName = getTypeName(tblgenType);
  std::string combined = ("!" + typeName).str();

  irdl::TypeOp op = builder.create<irdl::TypeOp>(
      UnknownLoc::get(ctx), StringAttr::get(ctx, combined));

  Block &opBlock = op.getBody().emplaceBlock();
  OpBuilder consBuilder = OpBuilder::atBlockBegin(&opBlock);

  if (tblgenType.hasSummary()) {
    auto summary = tblgenType.getSummary().trim();
    if (!summary.empty())
      consBuilder.create<irdl::SummaryOp>(
          UnknownLoc::get(ctx), StringAttr::get(ctx, tblgenType.getSummary()));
  }

  auto assemblyFormat = tblgenType.getAssemblyFormat();
  if (assemblyFormat.has_value())
    consBuilder.create<irdl::AssemblyFormatOp>(
        UnknownLoc::get(ctx),
        StringAttr::get(ctx, assemblyFormat.value().trim()));

  return op;
}

irdl::AttributeOp createIRDLAttr(OpBuilder &builder,
                                 tblgen::AttrDef &tblgenAttr) {
  MLIRContext *ctx = builder.getContext();
  StringRef attrName = getAttrName(tblgenAttr);
  std::string combined = ("#" + attrName).str();

  irdl::AttributeOp op = builder.create<irdl::AttributeOp>(
      UnknownLoc::get(ctx), StringAttr::get(ctx, combined));

  Block &opBlock = op.getBody().emplaceBlock();
  OpBuilder consBuilder = OpBuilder::atBlockBegin(&opBlock);

  if (tblgenAttr.hasSummary()) {
    auto summary = tblgenAttr.getSummary().trim();
    if (!summary.empty())
      consBuilder.create<irdl::SummaryOp>(
          UnknownLoc::get(ctx), StringAttr::get(ctx, tblgenAttr.getSummary()));
  }

  auto assemblyFormat = tblgenAttr.getAssemblyFormat();
  if (assemblyFormat.has_value())
    consBuilder.create<irdl::AssemblyFormatOp>(
        UnknownLoc::get(ctx),
        StringAttr::get(ctx, assemblyFormat.value().trim()));

  return op;
}

static irdl::DialectOp createIRDLDialect(OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  return builder.create<irdl::DialectOp>(UnknownLoc::get(ctx),
                                         StringAttr::get(ctx, selectedDialect));
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

  for (const Record *type :
       recordKeeper.getAllDerivedDefinitionsIfDefined("TypeDef")) {
    tblgen::TypeDef tblgenType(type);
    if (tblgenType.getDialect().getName() != selectedDialect)
      continue;
    createIRDLType(builder, tblgenType);
  }

  for (const Record *attr :
       recordKeeper.getAllDerivedDefinitionsIfDefined("AttrDef")) {
    tblgen::AttrDef tblgenAttr(attr);
    if (tblgenAttr.getDialect().getName() != selectedDialect)
      continue;
    createIRDLAttr(builder, tblgenAttr);
  }

  for (const Record *def :
       recordKeeper.getAllDerivedDefinitionsIfDefined("Op")) {
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
