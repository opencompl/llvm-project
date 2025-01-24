/*
{0}: TypeDef list
{1}: TypeParser function
{2}: TypePrinter function
{3}: Dialect CppName
{4}: TypeID Defines
{5}: Namespace open
{6}: Namespace close
*/

R"(

__NAMESPACE_OPEN__

{4}

class {1} : public ::mlir::Type::TypeBase<{1}, {3}, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "{2}.{0}";
  static constexpr ::llvm::StringLiteral dialectName = "{2}";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"{0}"};
  }
};

__NAMESPACE_CLOSE__

MLIR_DECLARE_EXPLICIT_TYPE_ID({6}::{1})
)"