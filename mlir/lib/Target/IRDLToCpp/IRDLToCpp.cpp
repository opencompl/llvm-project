//===- IRDLToCpp.cpp - Converts IRDL definitions to C++ -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/IRDLToCpp/IRDLToCpp.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "Templates/TemplatingUtils.h"

using namespace mlir;

constexpr char headerTemplateText[] =
#include "Templates/Header.txt"
    ;


// 0: Namespace open
// 1: Namespace close
// 2: Dialect C++ name
// 3: Dialect namespace
constexpr char dialectDefTemplateText[] =
#include "Templates/DialectDef.txt"
    ;

constexpr char declarationMacroFlag[] = "GEN_DIALECT_DECL_HEADER";
constexpr char definitionMacroFlag[] = "GEN_DIALECT_DEF";


constexpr char typeHeaderDefTemplateText[] =
#include "Templates/TypeHeaderDef.txt"
    ;


constexpr char typeDefTemplateText[] =
#include "Templates/TypeDef.txt"
    ;

constexpr char perOpDefTemplateText[] =
#include "Templates/PerOperationDef.txt"
    ;


namespace {

struct DialectStrings {
  StringRef dialectName;
  StringRef dialectCppName;
  StringRef dialectCppShortName;
  StringRef dialectBaseTypeName;

  StringRef namespaceOpen;
  StringRef namespaceClose;
  StringRef namespacePath;
};

struct TypeStrings {
  StringRef typeName;
  std::string typeCppName;
};

struct OpStrings {
  StringRef opName;
  std::string opCppName;
  llvm::SmallVector<std::string> opResultNames;
  llvm::SmallVector<std::string> opOperandNames;
};

static std::string capitalize(StringRef str) {
  return llvm::formatv("{0}{1}", llvm::toUpper(str[0]),
                       str.slice(1, str.size()));
}

static TypeStrings getStrings(irdl::TypeOp type) {
  TypeStrings strings;
  strings.typeName = type.getSymName();
  strings.typeCppName = llvm::formatv("{0}Type", capitalize(strings.typeName));
  return strings;
}

static OpStrings getStrings(irdl::OperationOp op) {

  auto &block = op.getBody().getBlocks().front();

  auto operands = block.getOps<irdl::OperandsOp>();
  auto operandOp =
      operands.empty() ? std::optional<irdl::OperandsOp>{} : *operands.begin();

  auto resultsOp = *block.getOps<irdl::ResultsOp>().begin();

  OpStrings strings;
  strings.opName = op.getSymName();
  strings.opCppName = llvm::formatv("{0}Op", capitalize(strings.opName));
  if (operandOp)
    strings.opOperandNames = llvm::SmallVector<std::string>(
        llvm::map_range(operandOp->getNames(), [](Attribute attr) {
          return llvm::formatv("{0}", cast<StringAttr>(attr));
        }));
  strings.opResultNames = llvm::SmallVector<std::string>(
      llvm::map_range(resultsOp.getNames(), [](Attribute attr) {
        return llvm::formatv("{0}", cast<StringAttr>(attr));
      }));

  return strings;
}

static void fillDict(irdl::detail::dictionary& dict, const TypeStrings& strings) {
  dict["TYPE_NAME"] = strings.typeName;
  dict["TYPE_CPP_NAME"] = strings.typeCppName;
}

static void fillDict(irdl::detail::dictionary& dict, const OpStrings& strings) {
  constexpr auto stringify = [](auto &&names) -> std::string {
    std::string nameArray;
    llvm::raw_string_ostream nameArrayStream(nameArray);
    nameArrayStream << "{\"" << llvm::join(names, "\", \"") << "\"}";

    return nameArray;
  };

  const auto operandCount = strings.opOperandNames.size();
  const auto resultCount = strings.opResultNames.size();

  dict["OP_NAME"] = strings.opName;
  dict["OP_CPP_NAME"] = strings.opCppName;
  dict["OP_OPERAND_COUNT"] = std::to_string(strings.opOperandNames.size());
  dict["OP_RESULT_COUNT"] = std::to_string(strings.opResultNames.size());
  dict["OP_OPERAND_INITIALIZER_LIST"] = operandCount ? stringify(strings.opOperandNames) : "{\"\"}";
  dict["OP_RESULT_INITIALIZER_LIST"] = resultCount ? stringify(strings.opResultNames) : "{\"\"}";

}


static void fillDict(irdl::detail::dictionary& dict, const DialectStrings& strings) {
  dict["DIALECT_NAME"] = strings.dialectName;
  dict["DIALECT_BASE_TYPE_NAME"] = strings.dialectBaseTypeName;
  dict["DIALECT_CPP_NAME"] = strings.dialectCppName;
  dict["DIALECT_CPP_SHORT_NAME"] = strings.dialectCppShortName;
  dict["NAMESPACE_OPEN"] = strings.namespaceOpen;
  dict["NAMESPACE_CLOSE"] = strings.namespaceClose;
  dict["NAMESPACE_PATH"] = strings.namespacePath;
}

static LogicalResult
generateTypedefList(mlir::Block &dialectBlock,
                    llvm::SmallVector<std::string> &typeNames) {
  auto typeOps = dialectBlock.getOps<irdl::TypeOp>();
  auto range = llvm::map_range(
      typeOps, [](auto &&type) { return getStrings(type).typeCppName; });
  typeNames = llvm::SmallVector<std::string>(range);
  return success();
}

static LogicalResult generateOpList(mlir::Block &dialectBlock,
                                    llvm::SmallVector<std::string> &typeNames) {
  auto typeOps = dialectBlock.getOps<irdl::OperationOp>();
  auto range = llvm::map_range(
      typeOps, [](auto &&type) { return getStrings(type).opCppName; });
  typeNames = llvm::SmallVector<std::string>(range);
  return success();
}

} // namespace

static LogicalResult generateTypeInclude(irdl::TypeOp type, raw_ostream &output,
                                         irdl::detail::dictionary &dict) {

  const auto typeDeclTemplate = irdl::detail::Template(
    #include "Templates/TypeDecl.txt"
  );

  const auto typeStrings = getStrings(type);
  fillDict(dict, typeStrings);

  typeDeclTemplate.render(output, dict);

  return success();
}

static LogicalResult generateOperationInclude(irdl::OperationOp op,
                                              raw_ostream &output,
                                              irdl::detail::dictionary &dict) {
  const auto perOpDeclTemplate = irdl::detail::Template(
    #include "Templates/PerOperationDecl.txt"
  ); 
  const auto opStrings = getStrings(op);
  fillDict(dict, opStrings);

  std::string tmp;
  llvm::raw_string_ostream stream {tmp};
  stream << llvm::formatv(
      R"(static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, {0} {1} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {{});)",
      llvm::join(llvm::map_range(opStrings.opResultNames,
                                 [](StringRef name) -> std::string {
                                   return llvm::formatv("::mlir::Type {0}, ",
                                                        name);
                                 }),
                 ""),
      llvm::join(llvm::map_range(opStrings.opOperandNames,
                                 [](StringRef name) -> std::string {
                                   return llvm::formatv("::mlir::Value {0}, ",
                                                        name);
                                 }),
                 ""));
  dict["OP_BUILD_DECLS"] = tmp;

  perOpDeclTemplate.render(output, dict);
  return success();
}

static LogicalResult generateInclude(irdl::DialectOp dialect,
                                     raw_ostream &output,
                                     DialectStrings &dialectStrings) {
  static const auto dialectDeclTemplate = irdl::detail::Template(
    #include "Templates/DialectDecl.txt"
  );
  static const auto typeHeaderDeclTemplate = irdl::detail::Template(
    #include "Templates/TypeHeaderDecl.txt"
  );
                                      
  output << "#ifdef " << declarationMacroFlag << "\n#undef "
         << declarationMacroFlag << "\n";

  irdl::detail::dictionary dict;
  fillDict(dict, dialectStrings);

  dialectDeclTemplate.render(output, dict);
  typeHeaderDeclTemplate.render(output, dict);

  auto &dialectBlock = *dialect.getRegion().getBlocks().begin();
  auto typeOps = dialectBlock.getOps<irdl::TypeOp>();
  auto operationOps = dialectBlock.getOps<irdl::OperationOp>();

  for (auto &&typeOp : typeOps) {
    if (failed(generateTypeInclude(typeOp, output, dict)))
      return failure();
  }

  llvm::SmallVector<std::string> opNames;
  if (failed(generateOpList(dialectBlock, opNames)))
    return failure();
  const auto forwardDeclarations = llvm::formatv(
      R"(
{1}
{0}
{2}
    )",
      llvm::join(llvm::map_range(opNames,
                                 [](llvm::StringRef name) -> std::string {
                                   return llvm::formatv("class {0};", name);
                                 }),
                 "\n"),
      dialectStrings.namespaceOpen, dialectStrings.namespaceClose);

  output << forwardDeclarations;
  for (auto &&operationOp : operationOps) {
    if (failed(generateOperationInclude(operationOp, output, dict)))
      return failure();
  }

  output << "#endif // " << declarationMacroFlag << "\n";

  return success();
}

static LogicalResult generateLib(irdl::DialectOp dialect, raw_ostream &output,
                                 DialectStrings &dialectStrings) {

  const auto opDefTemplateText = mlir::irdl::detail::Template{
  #include "Templates/OperationDef.txt"
  };

  output << "#ifdef " << definitionMacroFlag << "\n#undef "
         << definitionMacroFlag << "\n";

  // type header
  output << llvm::formatv(
      typeHeaderDefTemplateText, dialectStrings.dialectBaseTypeName,
      dialectStrings.dialectCppName, dialectStrings.namespaceOpen,
      dialectStrings.namespaceClose);

  output << "#endif // " << definitionMacroFlag << "\n";

  // get typedef list
  auto &dialectBlock = *dialect.getRegion().getBlocks().begin();
  llvm::SmallVector<std::string> typeNames;
  if (failed(generateTypedefList(dialectBlock, typeNames)))
    return failure();

  const auto commaSeparatedTypeList = llvm::join(
      llvm::map_range(typeNames,
                      [&dialectStrings](llvm::StringRef name) -> std::string {
                        return llvm::formatv(
                            "{0}::{1}", dialectStrings.namespacePath, name);
                      }),
      ",\n");

  const auto generatedTypeParser = llvm::formatv(
      R"(static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    {0}    
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {{
      *mnemonic = keyword;
      return std::nullopt;
    });
})",
      llvm::join(
          llvm::map_range(
              typeNames,
              [&](llvm::StringRef name) -> std::string {
                return llvm::formatv(
                    R"(.Case({1}::{0}::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = {1}::{0}::get(parser.getContext());
      return ::mlir::success(!!value);
    }))",
                    name, dialectStrings.namespacePath);
              }),
          "\n"));

  const auto generatedTypePrinter = llvm::formatv(
      R"(static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)
    {0}
    .Default([](auto) {{ return ::mlir::failure(); });
})",
      llvm::join(llvm::map_range(typeNames,
                                 [&](llvm::StringRef name) -> std::string {
                                   return llvm::formatv(
                                       R"(.Case<{1}::{0}>([&](auto t) {
      printer << {1}::{0}::getMnemonic();
      return ::mlir::success();
    }))",
                                       name, dialectStrings.namespacePath);
                                 }),
                 "\n"));

  const auto typeIdDefinitions =
      llvm::join(llvm::map_range(typeNames,
                                 [&](StringRef name) -> std::string {
                                   return llvm::formatv(
                                       "MLIR_DEFINE_EXPLICIT_TYPE_ID({1}::{0})",
                                       name, dialectStrings.namespacePath);
                                 }),
                 "\n");

  output << llvm::formatv(
      typeDefTemplateText, commaSeparatedTypeList, generatedTypeParser,
      generatedTypePrinter, dialectStrings.dialectCppName, typeIdDefinitions,
      dialectStrings.namespaceOpen, dialectStrings.namespaceClose);

  // get op list
  auto operations = dialectBlock.getOps<irdl::OperationOp>();
  llvm::SmallVector<std::string> opNames;
  if (failed(generateOpList(dialectBlock, opNames)))
    return failure();
  const auto commaSeparatedOpList = llvm::join(
      llvm::map_range(opNames,
                      [&dialectStrings](llvm::StringRef name) -> std::string {
                        return llvm::formatv(
                            "{0}::{1}", dialectStrings.namespacePath, name);
                      }),
      ",\n");

  const auto perOpDefinitions = llvm::join(
      llvm::map_range(
          operations,
          [&dialectStrings](irdl::OperationOp op) -> std::string {
            auto opStrings = getStrings(op);

            constexpr auto stringify = [](auto &&names) -> std::string {
              std::string nameArray;
              llvm::raw_string_ostream nameArrayStream(nameArray);
              nameArrayStream << "{\"" << llvm::join(names, "\", \"") << "\"}";

              return nameArray;
            };

            const auto operandCount = opStrings.opOperandNames.size();
            const auto operandNames =
                operandCount ? stringify(opStrings.opOperandNames) : "{\"\"}";

            const auto resultCount = opStrings.opResultNames.size();
            const auto resultNames = stringify(opStrings.opResultNames);

            const auto buildDefinition = llvm::formatv(
                R"(
void {0}::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, {1} {2} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {{
{3}
{4}
}
    )",
                opStrings.opCppName,
                llvm::join(llvm::map_range(opStrings.opResultNames,
                                           [](StringRef attr) -> std::string {
                                             return llvm::formatv(
                                                 "::mlir::Type {0}, ", attr);
                                           }),
                           ""),
                llvm::join(llvm::map_range(opStrings.opOperandNames,
                                           [](StringRef attr) -> std::string {
                                             return llvm::formatv(
                                                 "::mlir::Value {0}, ", attr);
                                           }),
                           ""),
                llvm::join(llvm::map_range(opStrings.opOperandNames,
                                           [](StringRef attr) -> std::string {
                                             return llvm::formatv(
                                                 "  odsState.addOperands({0});",
                                                 attr);
                                           }),
                           "\n"),
                llvm::join(llvm::map_range(opStrings.opResultNames,
                                           [](StringRef attr) -> std::string {
                                             return llvm::formatv(
                                                 "  odsState.addTypes({0});",
                                                 attr);
                                           }),
                           "\n"));
            return llvm::formatv(
                perOpDefTemplateText, opStrings.opCppName, operandCount,
                resultCount, buildDefinition, dialectStrings.namespaceOpen,
                dialectStrings.namespaceClose, dialectStrings.namespacePath);
          }),
      "\n");

  irdl::detail::dictionary dict;
  dict["OP_LIST"] = commaSeparatedOpList;
  dict["OP_CLASSES"] = perOpDefinitions;
  opDefTemplateText.render(output, dict); 

  output << llvm::formatv(dialectDefTemplateText, dialectStrings.namespaceOpen,
                          dialectStrings.namespaceClose,
                          dialectStrings.dialectCppName,
                          dialectStrings.namespacePath, commaSeparatedOpList,
                          commaSeparatedTypeList);

  output << "#endif // " << definitionMacroFlag << "\n";
  return success();
}

LogicalResult irdl::translateIRDLDialectToCpp(irdl::DialectOp dialect,
                                              raw_ostream &output) {
  const auto typeDefTempl = detail::Template(
#include "Templates/TypeDefTest.cpp"
  );

  StringRef dialectName = dialect.getSymName();

  // TODO: deal with no more constraints than the verifier allows.
  if (dialectName.size() < 1)
    return dialect->emitError("dialect name must be more than one character");
  if (!llvm::isAlpha(dialectName[0]))
    return dialect->emitError("dialect name must start with a letter");
  if (!llvm::all_of(dialectName,
                    [](char c) { return llvm::isAlnum(c) || c == '_'; }))
    return dialect->emitError(
        "dialect name must only contain letters, numbers or underscores");

  // TODO: allow more complex path.
  llvm::SmallVector<llvm::SmallString<8>> namespaceAbsolutePath{{"mlir"},
                                                                dialectName};
  std::string namespaceOpen;
  std::string namespaceClose;
  std::string namespacePath;
  llvm::raw_string_ostream namespaceOpenStream(namespaceOpen);
  llvm::raw_string_ostream namespaceCloseStream(namespaceClose);
  llvm::raw_string_ostream namespacePathStream(namespacePath);
  for (auto &pathElement : namespaceAbsolutePath) {
    namespaceOpenStream << "namespace " << pathElement << " {\n";
    namespaceCloseStream << "} // namespace " << pathElement << "\n";
    namespacePathStream << "::" << pathElement;
  }

  // TODO: allow control over C++ name.
  std::string cppShortName =
      llvm::formatv("{0}{1}", llvm::toUpper(dialectName[0]),
                    dialectName.slice(1, dialectName.size()));
  std::string dialectBaseTypeName = llvm::formatv("{0}Type", cppShortName);
  std::string cppName = llvm::formatv("{0}Dialect", cppShortName);

  DialectStrings dialectStrings;
  dialectStrings.dialectName = dialectName;
  dialectStrings.dialectBaseTypeName = dialectBaseTypeName;
  dialectStrings.dialectCppName = cppName;
  dialectStrings.dialectCppShortName = cppShortName;
  dialectStrings.namespaceOpen = namespaceOpen;
  dialectStrings.namespaceClose = namespaceClose;
  dialectStrings.namespacePath = namespacePath;

  output << headerTemplateText;


  if (failed(generateInclude(dialect, output, dialectStrings)))
    return failure();

  // if (failed(generateLib(dialect, output, dialectStrings)))
  //   return failure();

  // dict.set<detail::tv_index("SOME_NONSENSE")>("test");
  return success();
}
