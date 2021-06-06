//===- ExtensibleDialect.h - Extensible dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialects that can register new operations/types/attributes at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_EXTENSIBLEDIALECT_H
#define MLIR_IR_EXTENSIBLEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
class MLIRContext;
class DialectAsmPrinter;
class DialectAsmParser;
class ParseResult;
class OptionalParseResult;
class ExtensibleDialect;

namespace detail {
struct DynamicTypeStorage;
} // namespace detail

//===----------------------------------------------------------------------===//
// Dynamic type
//===----------------------------------------------------------------------===//

class DynamicType;

/// This is the definition of a dynamic type. It stores the parser,
/// the printer, and the verifier.
/// Each dynamic type instance refer to one instance of this class.
class DynamicTypeDefinition {
public:
  using VerifierFn = llvm::unique_function<LogicalResult(
      function_ref<InFlightDiagnostic()>, ArrayRef<Attribute>) const>;
  using ParserFn = llvm::unique_function<ParseResult(
      DialectAsmParser &parser,
      llvm::SmallVectorImpl<Attribute> &parsedAttributes) const>;
  using PrinterFn = llvm::unique_function<void(
      DialectAsmPrinter &printer, ArrayRef<Attribute> params) const>;

  static std::unique_ptr<DynamicTypeDefinition>
  get(llvm::StringRef name, Dialect *dialect, VerifierFn &&verifier);

  static std::unique_ptr<DynamicTypeDefinition>
  get(llvm::StringRef name, Dialect *dialect, VerifierFn &&verifier,
      ParserFn &&parser, PrinterFn &&printer);

  /// Check that the type parameters are valid.
  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<Attribute> params) const {
    return verifier(emitError, params);
  }

  /// Get the unique identifier associated with the concrete type.
  TypeID getTypeID() const { return typeID; }

  /// Get the MLIRContext in which the dynamic types are uniqued.
  MLIRContext &getContext() const { return *ctx; }

  /// Get the name of the type, in the format 'typename' and
  /// not 'dialectname.typename'.
  StringRef getName() const { return name; }

  /// Get the dialect defining the type.
  Dialect *getDialect() const { return dialect; }

private:
  DynamicTypeDefinition(llvm::StringRef name, Dialect *dialect,
                        VerifierFn &&verifier, ParserFn &&parser,
                        PrinterFn &&printer);

  /// This constructor should only be used when we need a pointer to
  /// the DynamicTypeDefinition in the verifier, the parser, or the printer.
  /// The verifier, parser, and printer need thus to be initialized after the
  /// constructor.
  DynamicTypeDefinition(Dialect *dialect, llvm::StringRef name);

  /// Register the concrete type in the type Uniquer.
  void registerInTypeUniquer();

  /// The name should be prefixed with the dialect name followed by '.'.
  std::string name;

  /// Dialect in which this type is defined.
  Dialect *dialect;

  /// Verifier for the type parameters.
  VerifierFn verifier;

  /// Parse the type parameters.
  ParserFn parser;

  /// Print the type parameters.
  PrinterFn printer;

  /// Unique identifier for the concrete type.
  TypeID typeID;

  /// Context in which the concrete types are uniqued.
  MLIRContext *ctx;

  friend ExtensibleDialect;
  friend DynamicType;
};

/// A type defined at runtime.
/// Each DynamicType instance represent a different dynamic type.
class DynamicType
    : public Type::TypeBase<DynamicType, Type, detail::DynamicTypeStorage> {
public:
  // Inherit Base constructors.
  using Base::Base;

  /// Get an instance of a dynamic type given a dynamic type definition and
  /// type parameters.
  /// This function does not call the type verifier.
  static DynamicType get(DynamicTypeDefinition *typeDef,
                         ArrayRef<Attribute> params = {});

  /// Get an instance of a dynamic type given a dynamic type definition and type
  /// parameters.
  /// This function also call the verifier to check if the parameters are valid.
  static DynamicType getChecked(function_ref<InFlightDiagnostic()> emitError,
                                DynamicTypeDefinition *typeDef,
                                ArrayRef<Attribute> params = {});

  /// Get the type definition of the concrete type.
  DynamicTypeDefinition *getTypeDef();

  /// Get the type parameters.
  ArrayRef<Attribute> getParams();

  /// Check if a type is a specific dynamic type.
  static bool isa(Type type, DynamicTypeDefinition *typeDef) {
    return type.getTypeID() == typeDef->getTypeID();
  }

  /// Check if a type is a dynamic type.
  static bool classof(Type type);

  /// Parse the dynamic type parameters and construct the type.
  /// The parameters are either empty, and nothing is parsed,
  /// or they are in the format '<>' or '<attr (,attr)*>'.
  static ParseResult parse(DialectAsmParser &parser,
                           DynamicTypeDefinition *typeDef,
                           DynamicType &parsedType);

  /// Print the dynamic type with the format
  /// 'type' or 'type<>' if there is no parameters, or 'type<attr (,attr)*>'.
  void print(DialectAsmPrinter &printer);
};

//===----------------------------------------------------------------------===//
// Dynamic operation
//===----------------------------------------------------------------------===//

/// The definition of a dynamic operation.
/// It contains the name of the operation, its owning dialect, a verifier,
/// a printer, and parser.
class DynamicOpDefinition {
public:
  static std::unique_ptr<DynamicOpDefinition>
  get(StringRef name, Dialect *dialect,
      AbstractOperation::VerifyInvariantsFn &&verifyFn);

  static std::unique_ptr<DynamicOpDefinition>
  get(StringRef name, Dialect *dialect,
      AbstractOperation::VerifyInvariantsFn &&verifyFn,
      AbstractOperation::ParseAssemblyFn &&parseFn,
      AbstractOperation::PrintAssemblyFn &&printFn);

  static std::unique_ptr<DynamicOpDefinition>
  get(StringRef name, Dialect *dialect,
      AbstractOperation::VerifyInvariantsFn &&verifyFn,
      AbstractOperation::ParseAssemblyFn &&parseFn,
      AbstractOperation::PrintAssemblyFn &&printFn,
      AbstractOperation::FoldHookFn &&foldHookFn,
      AbstractOperation::GetCanonicalizationPatternsFn
          &&getCanonicalizationPatternsFn);

  void setVerifyFn(AbstractOperation::VerifyInvariantsFn &&verify) {
    verifyFn = std::move(verify);
  }

  void setParseFn(AbstractOperation::ParseAssemblyFn &&parse) {
    parseFn = std::move(parse);
  }

  void setPrintFn(AbstractOperation::PrintAssemblyFn &&print) {
    printFn = std::move(print);
  }

  void setFoldHookFn(AbstractOperation::FoldHookFn &&foldHook) {
    foldHookFn = std::move(foldHook);
  }

  void setGetCanonicalizationPatternsFn(
      AbstractOperation::GetCanonicalizationPatternsFn
          &&getCanonicalizationPatterns) {
    getCanonicalizationPatternsFn = std::move(getCanonicalizationPatterns);
  }

private:
  DynamicOpDefinition(StringRef name, Dialect *dialect,
                      AbstractOperation::VerifyInvariantsFn &&verifyFn,
                      AbstractOperation::ParseAssemblyFn &&parseFn,
                      AbstractOperation::PrintAssemblyFn &&printFn,
                      AbstractOperation::FoldHookFn &&foldHookFn,
                      AbstractOperation::GetCanonicalizationPatternsFn
                          &&getCanonicalizationPatternsFn);

  /// Unique identifier for this operation.
  TypeID typeID;

  /// Name of the operation.
  /// The name is prefixed with the dialect name.
  std::string name;

  /// Dialect defining this operation.
  Dialect *dialect;

  AbstractOperation::VerifyInvariantsFn verifyFn;
  AbstractOperation::ParseAssemblyFn parseFn;
  AbstractOperation::PrintAssemblyFn printFn;
  AbstractOperation::FoldHookFn foldHookFn;
  AbstractOperation::GetCanonicalizationPatternsFn
      getCanonicalizationPatternsFn;

  friend ExtensibleDialect;
};

//===----------------------------------------------------------------------===//
// Extensible dialect
//===----------------------------------------------------------------------===//

/// A dialect that can be extended with new operations/types/attributes at
/// runtime.
class ExtensibleDialect : public mlir::Dialect {
public:
  ExtensibleDialect(StringRef name, MLIRContext *ctx, TypeID typeID);

  /// Add a new type defined at runtime to the dialect.
  void addDynamicType(std::unique_ptr<DynamicTypeDefinition> &&type);

  /// Add a new operation defined at runtime to the dialect.
  void addDynamicOp(std::unique_ptr<DynamicOpDefinition> &&type);

  /// Check if the dialect is an extensible dialect.
  static bool classof(const mlir::Dialect *dialect);

  /// Returns nullptr if the definition was not found.
  DynamicTypeDefinition *lookupTypeDefinition(StringRef name) const {
    auto it = nameToDynTypes.find(name);
    if (it == nameToDynTypes.end())
      return nullptr;
    return it->second;
  }

  /// Returns nullptr if the definition was not found.
  DynamicTypeDefinition *lookupTypeDefinition(TypeID id) const {
    auto it = dynTypes.find(id);
    if (it == dynTypes.end())
      return nullptr;
    return it->second.get();
  }

protected:
  /// Parse the dynamic type 'typeName' in the dialect 'dialect'.
  /// typename should not be prefixed with the dialect name.
  /// If the dynamic type does not exist, return no value.
  /// Otherwise, parse it, and return the parse result.
  /// If the parsing succeed, put the resulting type in 'resultType'.
  OptionalParseResult parseOptionalDynamicType(StringRef typeName,
                                               DialectAsmParser &parser,
                                               Type &resultType) const;

  /// If 'type' is a dynamic type, print it.
  /// Returns success if the type was printed, and failure if the type was not a
  /// dynamic type.
  static LogicalResult printIfDynamicType(Type type,
                                          DialectAsmPrinter &printer);

private:
  /// The set of all dynamic types registered.
  llvm::DenseMap<TypeID, std::unique_ptr<DynamicTypeDefinition>> dynTypes;

  /// This structure allows to get in O(1) a dynamic type given its name.
  llvm::StringMap<DynamicTypeDefinition *> nameToDynTypes;
};
} // namespace mlir

namespace llvm {
/// Provide isa functionality for ExtensibleDialect.
/// This is to override the isa functionality for Dialect.
template <>
struct isa_impl<mlir::ExtensibleDialect, mlir::Dialect> {
  static inline bool doit(const ::mlir::Dialect &dialect) {
    return mlir::ExtensibleDialect::classof(&dialect);
  }
};
} // namespace llvm

#endif // MLIR_IR_EXTENSIBLEDIALECT_H
