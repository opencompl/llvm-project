//===-- AttrTypeBase.td - Base Attr/Type definition file ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the base set of constructs for defining Attribute and
// Type classes.
//
//===----------------------------------------------------------------------===//

#ifndef ATTRTYPEBASE_TD
#define ATTRTYPEBASE_TD

include "mlir/IR/Constraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/Properties.td"
include "mlir/IR/Traits.td"

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

// A type, carries type constraints.
class Type<Pred condition, string descr = "",
           string cppClassName = "::mlir::Type"> :
    TypeConstraint<condition, descr, cppClassName> {
  string description = "";
  string builderCall = "";
}

// Allows providing an alternative name and summary to an existing type def.
class TypeAlias<Type t, string summary = t.summary> :
    Type<t.predicate, summary, t.cppClassName> {
  let description = t.description;
  let builderCall = t.builderCall;
}

// A type of a specific dialect.
class DialectType<Dialect d, Pred condition, string descr = "",
                  string cppClassName = "::mlir::Type"> :
    Type<condition, descr, cppClassName> {
  Dialect dialect = d;
}

// A variadic type constraint. It expands to zero or more of the base type. This
// class is used for supporting variadic operands/results.
class Variadic<Type type> : TypeConstraint<type.predicate, type.summary,
                                           type.cppClassName> {
  Type baseType = type;
  int minSize = 0;
}

// A nested variadic type constraint. It expands to zero or more variadic ranges
// of the base type. This class is used for supporting variadic operands and
// results. `variadicSegmentAttrName` should correspond to the name of an
// DenseI32ArrayAttr argument that provides the sizes of the inner variadic
// operand groups.
class VariadicOfVariadic<Type type, string variadicSegmentAttrName>
    : Variadic<type> {
  string segmentAttrName = variadicSegmentAttrName;
}

// An optional type constraint. It expands to either zero or one of the base
// type. This class is used for supporting optional operands/results.
class Optional<Type type> : TypeConstraint<type.predicate, type.summary,
                                           type.cppClassName> {
  Type baseType = type;
}

// A type that can be constructed using MLIR::Builder.
// Note that this does not "inherit" from Type because it would require
// duplicating Type subclasses for buildable and non-buildable cases to avoid
// diamond "inheritance".
// TODO: we may extend this to a more general 'Buildable' trait, making some
// Types and some Attrs buildable.
class BuildableType<code builder> {
  // The builder call to invoke (if specified) to construct the BuildableType.
  code builderCall = builder;
}

// A type that's buildable iff the type passed as an argument is buildable.
// This is intended for use by types like container types, which are only
// buildable if the type of their elements is buildable.
class SameBuildabilityAs<Type type, code builder> {
  code builderCall = !if(!empty(type.builderCall), "", builder);
}

// Any type at all.
def AnyType : Type<CPred<"true">, "any type">;

// None type
def NoneType : Type<CPred<"::llvm::isa<::mlir::NoneType>($_self)">, "none type",
                    "::mlir::NoneType">,
      BuildableType<"$_builder.getType<::mlir::NoneType>()">;

// Any type from the given list
class AnyTypeOf<list<Type> allowedTypes, string summary = "",
                string cppClassName = "::mlir::Type"> : Type<
    // Satisfy any of the allowed types' conditions.
    Or<!foreach(allowedtype, allowedTypes, allowedtype.predicate)>,
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedTypes, t.summary), " or "),
        summary),
    cppClassName>;

// A type that satisfies the constraints of all given types.
class AllOfType<list<Type> allowedTypes, string summary = "",
                string cppClassName = "::mlir::Type"> : Type<
    // Satisfy all of the allowedf types' conditions.
    And<!foreach(allowedType, allowedTypes, allowedType.predicate)>,
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedTypes, t.summary), " and "),
        summary),
    cppClassName>;

// A type that satisfies additional predicates.
class ConfinedType<Type type, list<Pred> predicates, string summary = "",
                   string cppClassName = type.cppClassName> : Type<
    And<!listconcat([type.predicate], !foreach(pred, predicates, pred))>,
    summary, cppClassName>;

// Integer types.

// Any integer type irrespective of its width and signedness semantics.
def AnyInteger : Type<CPred<"::llvm::isa<::mlir::IntegerType>($_self)">, "integer",
                      "::mlir::IntegerType">;

// Any integer type (regardless of signedness semantics) of a specific width.
class AnyI<int width>
    : Type<CPred<"$_self.isInteger(" # width # ")">, width # "-bit integer"> {
  int bitwidth = width;
}

class AnyIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, AnyI<w>),
              !interleave(widths, "/") # "-bit integer",
              "::mlir::IntegerType">;

def AnyI1  : AnyI<1>;
def AnyI8  : AnyI<8>;
def AnyI16 : AnyI<16>;
def AnyI32 : AnyI<32>;
def AnyI64 : AnyI<64>;

// Any signless integer type irrespective of its width.
def AnySignlessInteger : Type<
  CPred<"$_self.isSignlessInteger()">, "signless integer",
        "::mlir::IntegerType">;

// Signless integer type of a specific width.
class I<int width>
    : Type<CPred<"$_self.isSignlessInteger(" # width # ")">,
                  width # "-bit signless integer", "::mlir::IntegerType">,
      BuildableType<"$_builder.getIntegerType(" # width # ")"> {
  int bitwidth = width;
}

class SignlessIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, I<w>),
              !interleave(widths, "/") # "-bit signless integer">;

def I1  : I<1>;
def I8  : I<8>;
def I16 : I<16>;
def I32 : I<32>;
def I64 : I<64>;
def I128 : I<128>;

// Any signed integer type irrespective of its width.
def AnySignedInteger : Type<
  CPred<"$_self.isSignedInteger()">, "signed integer">;

// Signed integer type of a specific width.
class SI<int width>
    : Type<CPred<"$_self.isSignedInteger(" # width # ")">,
                  width # "-bit signed integer", "::mlir::IntegerType">,
      BuildableType<
        "$_builder.getIntegerType(" # width # ", /*isSigned=*/true)"> {
  int bitwidth = width;
}

class SignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, SI<w>),
              !interleave(widths, "/") # "-bit signed integer">;

def SI1  : SI<1>;
def SI8  : SI<8>;
def SI16 : SI<16>;
def SI32 : SI<32>;
def SI64 : SI<64>;

// Any unsigned integer type irrespective of its width.
def AnyUnsignedInteger : Type<
  CPred<"$_self.isUnsignedInteger()">, "unsigned integer">;

// Unsigned integer type of a specific width.
class UI<int width>
    : Type<CPred<"$_self.isUnsignedInteger(" # width # ")">,
                  width # "-bit unsigned integer", "::mlir::IntegerType">,
      BuildableType<
        "$_builder.getIntegerType(" # width # ", /*isSigned=*/false)"> {
  int bitwidth = width;
}

class UnsignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, UI<w>),
              !interleave(widths, "/") # "-bit unsigned integer">;

def UI1  : UI<1>;
def UI8  : UI<8>;
def UI16 : UI<16>;
def UI32 : UI<32>;
def UI64 : UI<64>;

// Index type.
def Index : Type<CPred<"::llvm::isa<::mlir::IndexType>($_self)">, "index",
                 "::mlir::IndexType">,
            BuildableType<"$_builder.getIndexType()">;

// Any signless integer type or index type.
def AnySignlessIntegerOrIndex : Type<CPred<"$_self.isSignlessIntOrIndex()">,
                                     "signless integer or index">;

// Floating point types.

// Any float type irrespective of its width.
def AnyFloat : Type<CPred<"::llvm::isa<::mlir::FloatType>($_self)">, "floating-point",
                    "::mlir::FloatType">;

// Float type of a specific width.
class F<int width>
    : Type<CPred<"$_self.isF" # width # "()">,
           width # "-bit float", "::mlir::FloatType">,
      BuildableType<"$_builder.getF" # width # "Type()"> {
  int bitwidth = width;
}

class FloatOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, F<w>),
              !interleave(widths, "/") # "-bit float">;

def F16 : F<16>;
def F32 : F<32>;
def F64 : F<64>;
def F80 : F<80>;
def F128 : F<128>;

def BF16 : Type<CPred<"$_self.isBF16()">, "bfloat16 type">,
           BuildableType<"$_builder.getBF16Type()">;
def TF32 : Type<CPred<"$_self.isTF32()">, "tf32 type">,
           BuildableType<"$_builder.getTF32Type()">;
def F8E4M3FN : Type<CPred<"$_self.isFloat8E4M3FN()">, "f8E4M3FN type">,
               BuildableType<"$_builder.getFloat8E4M3FNType()">;
def F8E5M2 : Type<CPred<"$_self.isFloat8E5M2()">, "f8E5M2 type">,
             BuildableType<"$_builder.getFloat8E5M2Type()">;
def F8E4M3FNUZ : Type<CPred<"$_self.isFloat8E4M3FNUZ()">, "f8E4M3FNUZ type">,
                 BuildableType<"$_builder.getFloat8E4M3FNUZType()">;
def F8E4M3B11FNUZ : Type<CPred<"$_self.isFloat8E4M3B11FNUZ()">, "f8E4M3B11FNUZ type">,
                 BuildableType<"$_builder.getFloat8E4M3B11FNUZType()">;
def F8E5M2FNUZ : Type<CPred<"$_self.isFloat8E5M2FNUZ()">, "f8E5M2FNUZ type">,
                 BuildableType<"$_builder.getFloat8E5M2FNUZType()">;

def AnyComplex : Type<CPred<"::llvm::isa<::mlir::ComplexType>($_self)">,
                      "complex-type", "::mlir::ComplexType">;

class Complex<Type type>
    : ConfinedType<AnyComplex, [
          SubstLeaves<"$_self",
                      "::llvm::cast<::mlir::ComplexType>($_self).getElementType()",
           type.predicate>],
           "complex type with " # type.summary # " elements",
           "::mlir::ComplexType">,
      SameBuildabilityAs<type, "::mlir::ComplexType::get($_builder.get" # type #
                               "Type())"> {
  Type elementType = type;
}

class OpaqueType<string dialect, string name, string summary>
  : Type<CPred<"isOpaqueTypeWithName($_self, \""#dialect#"\", \""#name#"\")">,
         summary, "::mlir::OpaqueType">,
    BuildableType<"::mlir::OpaqueType::get("
                  "$_builder.getStringAttr(\"" # dialect # "\"), \""
                  # name # "\")">;

// Function Type

// Any function type.
def FunctionType : Type<CPred<"::llvm::isa<::mlir::FunctionType>($_self)">,
                              "function type", "::mlir::FunctionType">;

// A container type is a type that has another type embedded within it.
class ContainerType<Type etype, Pred containerPred, code elementTypeCall,
                    string descr, string cppClassName = "::mlir::Type"> :
    // First, check the container predicate.  Then, substitute the extracted
    // element into the element type checker.
    Type<And<[containerPred,
                SubstLeaves<"$_self", !cast<string>(elementTypeCall),
                etype.predicate>]>,
         descr # " of " # etype.summary # " values", cppClassName>;

class ShapedContainerType<list<Type> allowedTypes,
                          Pred containerPred, string descr,
                          string cppClassName = "::mlir::Type"> :
    Type<And<[containerPred,
              Concat<"[](::mlir::Type elementType) { return ",
                SubstLeaves<"$_self", "elementType",
                AnyTypeOf<allowedTypes>.predicate>,
                "; }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())">]>,
         descr # " of " # AnyTypeOf<allowedTypes>.summary # " values", cppClassName>;

// Whether a shaped type is ranked.
def HasRankPred : CPred<"::llvm::cast<::mlir::ShapedType>($_self).hasRank()">;

// Whether a shaped type has one of the specified ranks.
class HasAnyRankOfPred<list<int> ranks> : And<[
    HasRankPred,
    Or<!foreach(rank, ranks,
                CPred<[{::llvm::cast<::mlir::ShapedType>($_self).getRank()
                         == }]
                      # rank>)>]>;

// Whether a shaped type has a rank greater than or equal of the specified rank.
class HasRankGreaterOrEqualPred<int rank> : And<[
    HasRankPred,
    CPred<[{::llvm::cast<::mlir::ShapedType>($_self).getRank() >= }] # rank>
]>;

// Vector types.

class VectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorTypePred, "vector",
                      "::mlir::VectorType">;

// Temporary vector type clone that allows gradual transition to 0-D vectors.
// TODO: Remove this when all ops support 0-D vectors.
class VectorOfAnyRankOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorOfAnyRankTypePred, "vector",
                      "::mlir::VectorType">;

class FixedVectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsFixedVectorTypePred,
          "fixed-length vector", "::mlir::VectorType">;

class ScalableVectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsScalableVectorTypePred,
          "scalable vector", "::mlir::VectorType">;

// Whether the number of elements of a vector is from the given
// `allowedRanks` list
class IsVectorOfRankPred<list<int> allowedRanks> :
  And<[IsVectorTypePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getRank()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a fixed-length vector is from the given
// `allowedRanks` list
class IsFixedVectorOfRankPred<list<int> allowedRanks> :
  And<[IsFixedVectorTypePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getRank()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a scalable vector is from the given
// `allowedRanks` list
class IsScalableVectorOfRankPred<list<int> allowedRanks> :
  And<[IsScalableVectorTypePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getRank()
                           == }]
                         # allowedlength>)>]>;

// Any vector where the rank is from the given `allowedRanks` list
class VectorOfRank<list<int> allowedRanks> : Type<
  IsVectorOfRankPred<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::VectorType">;

// Any fixed-length vector where the rank is from the given `allowedRanks` list
class FixedVectorOfRank<list<int> allowedRanks> : Type<
  IsFixedVectorOfRankPred<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::VectorType">;

// Any scalable vector where the rank is from the given `allowedRanks` list
class ScalableVectorOfRank<list<int> allowedRanks> : Type<
  IsScalableVectorOfRankPred<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::VectorType">;

// Any vector where the rank is from the given `allowedRanks` list and the type
// is from the given `allowedTypes` list
class VectorOfRankAndType<list<int> allowedRanks,
                          list<Type> allowedTypes> : AllOfType<
  [VectorOf<allowedTypes>, VectorOfRank<allowedRanks>],
  VectorOf<allowedTypes>.summary # VectorOfRank<allowedRanks>.summary,
  "::mlir::VectorType">;

// Whether the number of elements of a vector is from the given
// `allowedLengths` list
class IsVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getNumElements()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a fixed-length vector is from the given
// `allowedLengths` list
class IsFixedVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsFixedVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getNumElements()
                           == }]
                         # allowedlength>)>]>;

// Whether the number of elements of a scalable vector is from the given
// `allowedLengths` list
class IsScalableVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsScalableVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getNumElements()
                           == }]
                         # allowedlength>)>]>;

// Whether the shape of a vector matches the given `shape` list.
class IsVectorOfShape<list<int> shape>
  : CPred<"::llvm::cast<::mlir::VectorType>($_self).getShape() == ArrayRef<int64_t>({" # !interleave(shape, ", ") # "})">;

// Any vector where the number of elements is from the given
// `allowedLengths` list
class VectorOfLength<list<int> allowedLengths> : Type<
  IsVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"),
  "::mlir::VectorType">;

// Any fixed-length vector where the number of elements is from the given
// `allowedLengths` list
class FixedVectorOfLength<list<int> allowedLengths> : Type<
  IsFixedVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"),
  "::mlir::VectorType">;

// Any scalable vector where the number of elements is from the given
// `allowedLengths` list
class ScalableVectorOfLength<list<int> allowedLengths> : Type<
  IsScalableVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"),
  "::mlir::VectorType">;

// Any vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes`
// list
class VectorOfLengthAndType<list<int> allowedLengths,
                            list<Type> allowedTypes> : AllOfType<
  [VectorOf<allowedTypes>, VectorOfLength<allowedLengths>],
  VectorOf<allowedTypes>.summary # VectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

// Any fixed-length vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes` list
class FixedVectorOfLengthAndType<list<int> allowedLengths,
                                    list<Type> allowedTypes> : AllOfType<
  [FixedVectorOf<allowedTypes>, FixedVectorOfLength<allowedLengths>],
  FixedVectorOf<allowedTypes>.summary #
  FixedVectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

// Any scalable vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes` list
class ScalableVectorOfLengthAndType<list<int> allowedLengths,
                                    list<Type> allowedTypes> : AllOfType<
  [ScalableVectorOf<allowedTypes>, ScalableVectorOfLength<allowedLengths>],
  ScalableVectorOf<allowedTypes>.summary #
  ScalableVectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

def AnyVector : VectorOf<[AnyType]>;
// Temporary vector type clone that allows gradual transition to 0-D vectors.
def AnyVectorOfAnyRank : VectorOfAnyRankOf<[AnyType]>;

def AnyFixedVector : FixedVectorOf<[AnyType]>;

def AnyScalableVector : ScalableVectorOf<[AnyType]>;

// Shaped types.

def AnyShaped: ShapedContainerType<[AnyType], IsShapedTypePred, "shaped",
                                   "::mlir::ShapedType">;

//===----------------------------------------------------------------------===//
// Tensor types.

// Unranked tensor type whose element type is from the given `allowedTypes`
// list, and which additionally satisfies an optional list of predicates.
class UnrankedTensorOf<list<Type> allowedTypes, list<Pred> preds = [],
                       string summary = "unranked tensor">
  : ShapedContainerType<
      allowedTypes, And<!listconcat([IsUnrankedTensorTypePred], preds)>,
      summary, "::mlir::UnrankedTensorType">;

// Ranked tensor type whose element type is from the given `allowedTypes` list,
// and which additionally satisfies an optional list of predicates.
class RankedTensorOf<list<Type> allowedTypes, list<Pred> preds = [],
                     string summary = "ranked tensor">
  : ShapedContainerType<
      allowedTypes, And<!listconcat([IsRankedTensorTypePred], preds)>,
      summary, "::mlir::RankedTensorType">;

// Any tensor type whose element type is from the given `allowedTypes`
// list, and which additionally satisfies an optional list of predicates.
//
// TODO: use `Constraint` instead of `Pred`, so we can generate a better
// default summary (a la `ConfinedAttr`).
class TensorOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "tensor">
  : ShapedContainerType<allowedTypes,
      And<!listconcat([IsTensorTypePred], preds)>,
      summary, "::mlir::TensorType">;

def AnyTensor  : TensorOf<[AnyType]>;

def I1Tensor   : TensorOf<[I1]>;
def I8Tensor   : TensorOf<[I8]>;
def I16Tensor  : TensorOf<[I16]>;
def I32Tensor  : TensorOf<[I32]>;
def I64Tensor  : TensorOf<[I64]>;
def IndexTensor: TensorOf<[Index]>;

def BF16Tensor : TensorOf<[BF16]>;
def F16Tensor  : TensorOf<[F16]>;
def F32Tensor  : TensorOf<[F32]>;
def F64Tensor  : TensorOf<[F64]>;

class Non0RankedTensorOf<list<Type> allowedTypes>
  : TensorOf<allowedTypes, [HasRankGreaterOrEqualPred<1>],
      "non-0-ranked.tensor">;

def AnyRankedTensor : RankedTensorOf<[AnyType]>;
def AnyNon0RankedTensor  : Non0RankedTensorOf<[AnyType]>;
def AnyUnrankedTensor  : UnrankedTensorOf<[AnyType]>;

def AnyNon0RankedOrUnrankedTensor
  : AnyTypeOf<[AnyUnrankedTensor, AnyNon0RankedTensor],
              "non-0-ranked or unranked tensor", "::mlir::TensorType">;

// Ranked tensor type with one of the specified types and ranks.
class TensorRankOf<list<Type> allowedTypes, list<int> ranks>
  : RankedTensorOf<allowedTypes,
      [HasAnyRankOfPred<ranks>],
      !interleave(!foreach(rank, ranks, rank # "D"), "/") # " tensor">;

class 0DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [0]>;
class 1DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [1]>;
class 2DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [2]>;
class 3DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [3]>;
class 4DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [4]>;

class StaticShapeTensorOf<list<Type> allowedTypes>
  : RankedTensorOf<allowedTypes, [HasStaticShapePred],
                   "statically shaped tensor">;

def AnyStaticShapeTensor : StaticShapeTensorOf<[AnyType]>;

//===----------------------------------------------------------------------===//
// Memref type.

// Any unranked memref whose element type is from the given `allowedTypes` list.
class UnrankedMemRefOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes,
                        IsUnrankedMemRefTypePred, "unranked.memref",
                        "::mlir::UnrankedMemRefType">;

def AnyUnrankedMemRef : UnrankedMemRefOf<[AnyType]>;

// Any ranked memref whose element type is from the given `allowedTypes` list.
class MemRefOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes, IsMemRefTypePred, "memref",
                        "::mlir::MemRefType">;

class Non0RankedMemRefOf<list<Type> allowedTypes> :
    ConfinedType<MemRefOf<allowedTypes>, [HasRankGreaterOrEqualPred<1>],
         "non-0-ranked." # MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

def AnyMemRef : MemRefOf<[AnyType]>;
def AnyNon0RankedMemRef : Non0RankedMemRefOf<[AnyType]>;

// Any memref (ranked or unranked) whose element type is from the given
// `allowedTypes` list, and which additionally satisfies an optional list of
// predicates.
class RankedOrUnrankedMemRefOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "ranked or unranked memref">
  : ShapedContainerType<allowedTypes,
      And<!listconcat([IsBaseMemRefTypePred], preds)>,
      summary, "::mlir::BaseMemRefType">;

def AnyRankedOrUnrankedMemRef  : RankedOrUnrankedMemRefOf<[AnyType]>;
def AnyNon0RankedOrUnrankedMemRef:
    AnyTypeOf<[AnyUnrankedMemRef, AnyNon0RankedMemRef]>;

// Memref declarations handle any memref, independent of rank, size, (static or
// dynamic), layout, or memory space.
def I1MemRef  : MemRefOf<[I1]>;
def I8MemRef  : MemRefOf<[I8]>;
def I16MemRef : MemRefOf<[I16]>;
def I32MemRef : MemRefOf<[I32]>;
def I64MemRef : MemRefOf<[I64]>;

def BF16MemRef : MemRefOf<[BF16]>;
def F16MemRef  : MemRefOf<[F16]>;
def F32MemRef  : MemRefOf<[F32]>;
def F64MemRef  : MemRefOf<[F64]>;

// TODO: Have an easy way to add another constraint to a type.
class MemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    ConfinedType<MemRefOf<allowedTypes>, [HasAnyRankOfPred<ranks>],
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

class StaticShapeMemRefOf<list<Type> allowedTypes> :
    ConfinedType<MemRefOf<allowedTypes>, [HasStaticShapePred],
         "statically shaped " # MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

def AnyStaticShapeMemRef : StaticShapeMemRefOf<[AnyType]>;

// For a MemRefType, verify that it has strides.
def HasStridesPred : CPred<[{ isStrided(::llvm::cast<::mlir::MemRefType>($_self)) }]>;

class StridedMemRefOf<list<Type> allowedTypes> :
    ConfinedType<MemRefOf<allowedTypes>, [HasStridesPred],
         "strided " # MemRefOf<allowedTypes>.summary>;

def AnyStridedMemRef : StridedMemRefOf<[AnyType]>;

class AnyStridedMemRefOfRank<int rank> :
  AllOfType<[AnyStridedMemRef, MemRefRankOf<[AnyType], [rank]>],
       AnyStridedMemRef.summary # " of rank " # rank>;

class StridedMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    ConfinedType<MemRefOf<allowedTypes>, [HasAnyRankOfPred<ranks>],
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary>;

// This represents a generic tuple without any constraints on element type.
def AnyTuple : Type<IsTupleTypePred, "tuple", "::mlir::TupleType">;

// A container type that has other types embedded in it, but (unlike
// ContainerType) can hold elements with a mix of types. Requires a call that
// produces a list of all elements' types.
class MixedContainerType<Type etype, Pred containerPred, code elementTypesCall,
                         string descr> :
    Type<
        And<[
            containerPred,
            Concat<
                "::llvm::all_of(" # elementTypesCall # ", [](::mlir::Type t) { "
                "return t && (",
                SubstLeaves<"$_self", "t", etype.predicate>,
                "); })"
            >
        ]>,
        descr # " with any combination of " # etype.summary # " values"> {
  // The type of elements in the container.
  Type elementType = etype;

  // Call to retrieve.
  code getElementTypesCall = elementTypesCall;
}

// A Tuple that holds a mix of elements of the allowed types.
class TupleOf<list<Type> allowedTypes>
    : MixedContainerType<AnyTypeOf<allowedTypes>, IsTupleTypePred,
                         "::llvm::cast<::mlir::TupleType>($_self).getTypes()",
                         "tuple">;

// A Tuple with arbitrary nesting, where all elements are a mix of the allowed
// types.
class NestedTupleOf<list<Type> allowedTypes> :
    MixedContainerType<AnyTypeOf<allowedTypes>, IsTupleTypePred,
                       "getFlattenedTypes(::llvm::cast<::mlir::TupleType>($_self))",
                       "nested tuple">;

//===----------------------------------------------------------------------===//
// Common type constraints
//===----------------------------------------------------------------------===//
// Type constraint for types that are "like" some type or set of types T, that is
// they're either a T, a vector of Ts, or a tensor of Ts
class TypeOrContainer<Type allowedType, string name> : TypeConstraint<Or<[
  allowedType.predicate, VectorOf<[allowedType]>.predicate,
  TensorOf<[allowedType]>.predicate]>,
  name>;

// Temporary constraint to allow gradual transition to supporting 0-D vectors.
// TODO: Remove this when all ops support 0-D vectors.
class TypeOrContainerOfAnyRank<Type allowedType, string name> : TypeConstraint<Or<[
  allowedType.predicate, VectorOfAnyRankOf<[allowedType]>.predicate,
  TensorOf<[allowedType]>.predicate]>,
  name>;


// Type constraint for bool-like types: bools, vectors of bools, tensors of
// bools.
def BoolLike : TypeOrContainer<I1, "bool-like">;

def BoolLikeOfAnyRank : TypeOrContainerOfAnyRank<I1, "bool-like">;

// Type constraint for signless-integer-like types: signless integers, indices,
// vectors of signless integers or indices, tensors of signless integers.
def SignlessIntegerLike : TypeOrContainer<AnySignlessIntegerOrIndex,
    "signless-integer-like">;

def SignlessIntegerLikeOfAnyRank : TypeOrContainerOfAnyRank<
    AnySignlessIntegerOrIndex,
    "signless-integer-like">;

// Type constraint for float-like types: floats, vectors or tensors thereof.
def FloatLike : TypeOrContainer<AnyFloat, "floating-point-like">;

// Type constraint for signless-integer-like or float-like types.
def SignlessIntegerOrFloatLike : TypeConstraint<Or<[
    SignlessIntegerLike.predicate, FloatLike.predicate]>,
    "signless-integer-like or floating-point-like">;

//===----------------------------------------------------------------------===//
// Attribute definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base attribute definition

// Base class for all attributes.
class Attr<Pred condition, string summary = ""> :
    AttrConstraint<condition, summary> {
  code storageType = ?; // The backing mlir::Attribute type
  code returnType = ?;  // The underlying C++ value type

  // The call expression to convert from the storage type to the return
  // type. For example, an enum can be stored as an int but returned as an
  // enum class.
  //
  // Format: $_self will be expanded to the attribute.
  //
  // For example, `$_self.getValue().getSExtValue()` for `IntegerAttr val` will
  // expand to `getAttrOfType<IntegerAttr>("val").getValue().getSExtValue()`.
  code convertFromStorage = "$_self.getValue()";

  // The call expression to build an attribute from a constant value.
  //
  // Format: $0 will be expanded to the constant value of the attribute.
  //
  // For example, `$_builder.getStringAttr("$0")` for `StringAttr:"foo"` will
  // expand to `builder.getStringAttr("foo")`.
  string constBuilderCall = ?;

  // Default value for attribute.
  // Requires a constBuilderCall defined.
  string defaultValue = ?;

  // The value type of this attribute. This corresponds to the mlir::Type that
  // this attribute returns via `getType()`.
  Type valueType = ?;

  // Whether the attribute is optional. Typically requires a custom
  // convertFromStorage method to handle the case where the attribute is
  // not present.
  bit isOptional = 0;

  // What is the base-level Attr instantiation that this Attr is built upon.
  // Unset means this is a base-level Attr.
  //
  // This field is used by attribute wrapper classes (DefaultValuedAttr,
  // OptionalAttr, etc.) to retrieve the base-level attribute definition.
  // This can be used for getting its name; otherwise, we will see
  // "anonymous_<number>" as the attribute def name because of template
  // instantiation.
  // TOOD(b/132458159): deduplicate the fields in attribute wrapper classes.
  Attr baseAttr = ?;

  // The fully-qualified C++ namespace where the generated class lives.
  string cppNamespace = "";

  // The full description of this attribute.
  string description = "";
}

// An attribute of a specific dialect.
class DialectAttr<Dialect d, Pred condition, string summary = ""> :
    Attr<condition, summary> {
  Dialect dialect = d;
  let cppNamespace = d.cppNamespace;
}

//===----------------------------------------------------------------------===//
// Attribute modifier definition

// Decorates an attribute to have an (unvalidated) default value if not present.
class DefaultValuedAttr<Attr attr, string val> :
    Attr<attr.predicate, attr.summary> {
  // Construct this attribute with the input attribute and change only
  // the default value.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = val;
  let valueType = attr.valueType;

  let baseAttr = attr;
}

// Decorates an optional attribute to have an (unvalidated) default value
// return by ODS generated accessors if not present.
class DefaultValuedOptionalAttr<Attr attr, string val> :
    Attr<attr.predicate, attr.summary> {
  // Construct this attribute with the input attribute and change only
  // the default value.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = val;
  let valueType = attr.valueType;
  let isOptional = 1;

  let baseAttr = attr;
}

// Decorates an attribute as optional. The return type of the generated
// attribute accessor method will be Optional<>.
class OptionalAttr<Attr attr> : Attr<attr.predicate, attr.summary> {
  // Rewrite the attribute to be optional.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = "::std::optional<" # attr.returnType #">";
  let convertFromStorage = "$_self ? " # returnType # "(" #
                           attr.convertFromStorage # ") : (::std::nullopt)";
  let valueType = attr.valueType;
  let isOptional = 1;

  let baseAttr = attr;
}

// Default-valued string-based attribute. Wraps the default value in escaped
// quotes.
class DefaultValuedStrAttr<Attr attr, string val>
    : DefaultValuedAttr<attr, "\"" # val # "\"">;
class DefaultValuedOptionalStrAttr<Attr attr, string val>
    : DefaultValuedOptionalAttr<attr, "\"" # val # "\"">;

//===----------------------------------------------------------------------===//
// Primitive property kinds

// Any kind of integer stored as properties.
class IntProperty<string storageTypeParam = "", string desc = ""> :
    Property<storageTypeParam, desc> {
  code writeToMlirBytecode = [{
    $_writer.writeVarInt($_storage);
  }];
  code readFromMlirBytecode = [{
    uint64_t val;
    if (failed($_reader.readVarInt(val)))
      return ::mlir::failure();
    $_storage = val;
  }];
}

class ArrayProperty<string storageTypeParam = "", int n, string desc = ""> :
  Property<storageTypeParam # "[" # n # "]", desc> {
  let interfaceType = "::llvm::ArrayRef<" # storageTypeParam # ">";
  let convertFromStorage = "$_storage";
  let assignToStorage = "::llvm::copy($_value, $_storage)";
}

//===----------------------------------------------------------------------===//
// Primitive attribute kinds

// A generic attribute that must be constructed around a specific buildable type
// `attrValType`. Backed by MLIR attribute kind `attrKind`.
class TypedAttrBase<Type attrValType, string attrKind, Pred condition,
                    string descr> :
    Attr<condition, descr> {
  let constBuilderCall = "$_builder.get" # attrKind # "(" #
                         attrValType.builderCall # ", $0)";
  let storageType = "::mlir::" # attrKind;
  let valueType = attrValType;
}

// Any attribute.
def AnyAttr : Attr<CPred<"true">, "any attribute"> {
  let storageType = "::mlir::Attribute";
  let returnType = "::mlir::Attribute";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

// Any attribute from the given list
class AnyAttrOf<list<Attr> allowedAttrs, string summary = "",
                string cppClassName = "::mlir::Attribute",
                string fromStorage = "$_self"> : Attr<
    // Satisfy any of the allowed attribute's condition
    Or<!foreach(allowedattr, allowedAttrs, allowedattr.predicate)>,
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedAttrs, t.summary), " or "),
        summary)> {
    let returnType = cppClassName;
    let convertFromStorage = fromStorage;
}

def LocationAttr : Attr<CPred<"::llvm::isa<::mlir::LocationAttr>($_self)">,
                        "location attribute">;

def BoolAttr : Attr<CPred<"::llvm::isa<::mlir::BoolAttr>($_self)">, "bool attribute"> {
  let storageType = [{ ::mlir::BoolAttr }];
  let returnType = [{ bool }];
  let valueType = I1;
  let constBuilderCall = "$_builder.getBoolAttr($0)";
}

// Index attribute.
def IndexAttr :
    TypedAttrBase<
      Index, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::isa<::mlir::IndexType>(::llvm::cast<::mlir::IntegerAttr>($_self).getType())">]>,
      "index attribute"> {
  let returnType = [{ ::llvm::APInt }];
}

// Base class for any integer (regardless of signedness semantics) attributes
// of fixed width.
class AnyIntegerAttrBase<AnyI attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getType()."
                 "isInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
  let constBuilderCall = ?;
}

def AnyI1Attr  : AnyIntegerAttrBase<AnyI1,  "1-bit integer attribute">;
def AnyI8Attr  : AnyIntegerAttrBase<AnyI8,  "8-bit integer attribute">;
def AnyI16Attr : AnyIntegerAttrBase<AnyI16, "16-bit integer attribute">;
def AnyI32Attr : AnyIntegerAttrBase<AnyI32, "32-bit integer attribute">;
def AnyI64Attr : AnyIntegerAttrBase<AnyI64, "64-bit integer attribute">;

def APIntAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
                     "arbitrary integer attribute"> {
  let storageType = [{ ::mlir::IntegerAttr }];
  let returnType = [{ ::mlir::APInt }];
}

// Base class for signless integer attributes of fixed width.
class SignlessIntegerAttrBase<I attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getType()."
                 "isSignlessInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
}
// Base class for signless integer attributes of fixed width that have a
// corresponding C++ type.
class TypedSignlessIntegerAttrBase<I attrValType, string retType, string descr>
    : SignlessIntegerAttrBase<attrValType, descr> {
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().getZExtValue()";
}

def I1Attr  : TypedSignlessIntegerAttrBase<
    I1,  "bool",     "1-bit signless integer attribute">;
def I8Attr  : TypedSignlessIntegerAttrBase<
    I8,  "uint8_t",  "8-bit signless integer attribute">;
def I16Attr : TypedSignlessIntegerAttrBase<
    I16, "uint16_t", "16-bit signless integer attribute">;
def I32Attr : TypedSignlessIntegerAttrBase<
    I32, "uint32_t", "32-bit signless integer attribute">;
def I64Attr : TypedSignlessIntegerAttrBase<
    I64, "uint64_t", "64-bit signless integer attribute">;

// Base class for signed integer attributes of fixed width.
class SignedIntegerAttrBase<SI attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getType()."
                 "isSignedInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
}
// Base class for signed integer attributes of fixed width that have a
// corresponding C++ type.
class TypedSignedIntegerAttrBase<SI attrValType, string retType, string descr>
    : SignedIntegerAttrBase<attrValType, descr> {
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().getSExtValue()";
}

def SI1Attr  : TypedSignedIntegerAttrBase<
    SI1,  "bool",    "1-bit signed integer attribute">;
def SI8Attr  : TypedSignedIntegerAttrBase<
    SI8,  "int8_t",  "8-bit signed integer attribute">;
def SI16Attr : TypedSignedIntegerAttrBase<
    SI16, "int16_t", "16-bit signed integer attribute">;
def SI32Attr : TypedSignedIntegerAttrBase<
    SI32, "int32_t", "32-bit signed integer attribute">;
def SI64Attr : TypedSignedIntegerAttrBase<
    SI64, "int64_t", "64-bit signed integer attribute">;

// Base class for unsigned integer attributes of fixed width.
class UnsignedIntegerAttrBase<UI attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getType()."
                 "isUnsignedInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
}
// Base class for unsigned integer attributes of fixed width that have a
// corresponding C++ type.
class TypedUnsignedIntegerAttrBase<UI attrValType, string retType, string descr>
    : UnsignedIntegerAttrBase<attrValType, descr> {
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().getZExtValue()";
}

def UI1Attr  : TypedUnsignedIntegerAttrBase<
    UI1,  "bool",     "1-bit unsigned integer attribute">;
def UI8Attr  : TypedUnsignedIntegerAttrBase<
    UI8,  "uint8_t",  "8-bit unsigned integer attribute">;
def UI16Attr : TypedUnsignedIntegerAttrBase<
    UI16, "uint16_t", "16-bit unsigned integer attribute">;
def UI32Attr : TypedUnsignedIntegerAttrBase<
    UI32, "uint32_t", "32-bit unsigned integer attribute">;
def UI64Attr : TypedUnsignedIntegerAttrBase<
    UI64, "uint64_t", "64-bit unsigned integer attribute">;

// Base class for float attributes of fixed width.
class FloatAttrBase<F attrValType, string descr> :
    TypedAttrBase<attrValType, "FloatAttr",
              And<[CPred<"::llvm::isa<::mlir::FloatAttr>($_self)">,
                     CPred<"::llvm::cast<::mlir::FloatAttr>($_self).getType().isF" #
                           attrValType.bitwidth # "()">]>,
              descr> {
  let returnType = [{ ::llvm::APFloat }];
}

def F32Attr : FloatAttrBase<F32, "32-bit float attribute">;
def F64Attr : FloatAttrBase<F64, "64-bit float attribute">;

// An attribute backed by a string type.
class StringBasedAttr<Pred condition, string descr> : Attr<condition, descr> {
  let constBuilderCall = "$_builder.getStringAttr($0)";
  let storageType = [{ ::mlir::StringAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
}

def StrAttr : StringBasedAttr<CPred<"::llvm::isa<::mlir::StringAttr>($_self)">,
                              "string attribute">;

// A string attribute that represents the name of a symbol.
def SymbolNameAttr : StringBasedAttr<CPred<"::llvm::isa<::mlir::StringAttr>($_self)">,
                                     "string attribute">;

// String attribute that has a specific value type.
class TypedStrAttr<Type ty>
    : StringBasedAttr<CPred<"::llvm::isa<::mlir::StringAttr>($_self)">,
                            "string attribute"> {
  let valueType = ty;
}

// Base class for attributes containing types. Example:
//   def IntTypeAttr : TypeAttrBase<"IntegerType", "integer type attribute">
// defines a type attribute containing an integer type.
class TypeAttrBase<string retType, string summary,
                        Pred typePred = CPred<"true">> :
    Attr<And<[
      CPred<"::llvm::isa<::mlir::TypeAttr>($_self)">,
      CPred<"::llvm::isa<" # retType # ">(::llvm::cast<::mlir::TypeAttr>($_self).getValue())">,
      SubstLeaves<"$_self",
                    "::llvm::cast<::mlir::TypeAttr>($_self).getValue()", typePred>]>,
    summary> {
  let storageType = [{ ::mlir::TypeAttr }];
  let returnType = retType;
  let valueType = NoneType;
  let convertFromStorage = "::llvm::cast<" # retType # ">($_self.getValue())";
}

def TypeAttr : TypeAttrBase<"::mlir::Type", "any type attribute"> {
  let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

class TypeAttrOf<Type ty>
   : TypeAttrBase<ty.cppClassName, "type attribute of " # ty.summary,
                    ty.predicate> {
  let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

// The mere presence of unit attributes has a meaning.  Therefore, unit
// attributes are always treated as optional and accessors to them return
// "true" if the attribute is present and "false" otherwise.
def UnitAttr : Attr<CPred<"::llvm::isa<::mlir::UnitAttr>($_self)">, "unit attribute"> {
  let storageType = [{ ::mlir::UnitAttr }];
  let constBuilderCall = "(($0) ? $_builder.getUnitAttr() : nullptr)";
  let convertFromStorage = "$_self != nullptr";
  let returnType = "bool";
  let defaultValue = "false";
  let valueType = NoneType;
  let isOptional = 1;
}

//===----------------------------------------------------------------------===//
// Composite attribute kinds

class DictionaryAttrBase<Pred condition, string summary> :
    Attr<condition, summary> {
  let storageType = [{ ::mlir::DictionaryAttr }];
  let constBuilderCall = "$_builder.getDictionaryAttr($0)";
  let returnType = [{ ::mlir::DictionaryAttr }];
  let valueType = NoneType;
  let convertFromStorage = "$_self";
}

def DictionaryAttr
    : DictionaryAttrBase<CPred<"::llvm::isa<::mlir::DictionaryAttr>($_self)">,
                               "dictionary of named attribute values">;

class ElementsAttrBase<Pred condition, string summary> :
    Attr<condition, summary> {
  let storageType = [{ ::mlir::ElementsAttr }];
  let returnType = [{ ::mlir::ElementsAttr }];
  let convertFromStorage = "$_self";
}

def ElementsAttr : ElementsAttrBase<CPred<"::llvm::isa<::mlir::ElementsAttr>($_self)">,
                                    "constant vector/tensor attribute">;

class IntElementsAttrBase<Pred condition, string summary> :
    ElementsAttrBase<And<[CPred<"::llvm::isa<::mlir::DenseIntElementsAttr>($_self)">,
                          condition]>,
                     summary> {
  let storageType = [{ ::mlir::DenseIntElementsAttr }];
  let returnType = [{ ::mlir::DenseIntElementsAttr }];

  let convertFromStorage = "$_self";
}

class DenseArrayAttrBase<string denseAttrName, string cppType, string summaryName> :
    ElementsAttrBase<CPred<"::llvm::isa<::mlir::" # denseAttrName # ">($_self)">,
                     summaryName # " dense array attribute"> {
  let storageType = "::mlir::" # denseAttrName;
  let returnType = "::llvm::ArrayRef<" # cppType # ">";
  let constBuilderCall = "$_builder.get" # denseAttrName # "($0)";
}
def DenseBoolArrayAttr : DenseArrayAttrBase<"DenseBoolArrayAttr", "bool", "i1">;
def DenseI8ArrayAttr : DenseArrayAttrBase<"DenseI8ArrayAttr", "int8_t", "i8">;
def DenseI16ArrayAttr : DenseArrayAttrBase<"DenseI16ArrayAttr", "int16_t", "i16">;
def DenseI32ArrayAttr : DenseArrayAttrBase<"DenseI32ArrayAttr", "int32_t", "i32">;
def DenseI64ArrayAttr : DenseArrayAttrBase<"DenseI64ArrayAttr", "int64_t", "i64">;
def DenseF32ArrayAttr : DenseArrayAttrBase<"DenseF32ArrayAttr", "float", "f32">;
def DenseF64ArrayAttr : DenseArrayAttrBase<"DenseF64ArrayAttr", "double", "f64">;

def IndexElementsAttr
    : IntElementsAttrBase<CPred<[{::llvm::cast<::mlir::DenseIntElementsAttr>($_self)
                                      .getType()
                                      .getElementType()
                                      .isIndex()}]>,
                          "index elements attribute">;

def AnyIntElementsAttr : IntElementsAttrBase<CPred<"true">, "integer elements attribute">;

class IntElementsAttrOf<int width> : IntElementsAttrBase<
  CPred<"::llvm::cast<::mlir::DenseIntElementsAttr>($_self).getType()."
        "getElementType().isInteger(" # width # ")">,
  width # "-bit integer elements attribute">;

def AnyI32ElementsAttr : IntElementsAttrOf<32>;
def AnyI64ElementsAttr : IntElementsAttrOf<64>;

class SignlessIntElementsAttr<int width> : IntElementsAttrBase<
  CPred<"::llvm::cast<::mlir::DenseIntElementsAttr>($_self).getType()."
        "getElementType().isSignlessInteger(" # width # ")">,
  width # "-bit signless integer elements attribute"> {

  // Note that this is only constructing scalar elements attribute.
  let constBuilderCall = "::llvm::cast<::mlir::DenseIntElementsAttr>("
  "::mlir::DenseElementsAttr::get("
    "::mlir::RankedTensorType::get({}, $_builder.getIntegerType(" # width # ")), "
    "::llvm::ArrayRef($0)))";
}

def I32ElementsAttr : SignlessIntElementsAttr<32>;
def I64ElementsAttr : SignlessIntElementsAttr<64>;

// A `width`-bit signless integer elements attribute. The attribute should be
// ranked and has a shape as specified in `dims`.
class RankedSignlessIntElementsAttr<int width, list<int> dims> :
    SignlessIntElementsAttr<width> {
  // Check that this has the specified shape.
  let predicate = And<[
    SignlessIntElementsAttr<width>.predicate,
    CPred<"::llvm::cast<::mlir::DenseIntElementsAttr>($_self).getType().getShape() == "
        "::mlir::ArrayRef<int64_t>({" # !interleave(dims, ", ") # "})">]>;

  let summary = width # "-bit signless int elements attribute of shape [" #
                !interleave(dims, ", ") # "]";

  let constBuilderCall = "::mlir::DenseIntElementsAttr::get("
    "::mlir::RankedTensorType::get({" # !interleave(dims, ", ") #
    "}, $_builder.getIntegerType(" # width # ")), ::llvm::ArrayRef($0))";
}

class RankedI32ElementsAttr<list<int> dims> :
    RankedSignlessIntElementsAttr<32, dims>;
class RankedI64ElementsAttr<list<int> dims> :
    RankedSignlessIntElementsAttr<64, dims>;

class FloatElementsAttr<int width> : ElementsAttrBase<
  CPred<"::llvm::isa<::mlir::DenseFPElementsAttr>($_self) &&"
      "::llvm::cast<::mlir::DenseElementsAttr>($_self).getType()."
      "getElementType().isF" # width # "()">,
  width # "-bit float elements attribute"> {

  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  // Note that this is only constructing scalar elements attribute.
  let constBuilderCall = "::mlir::DenseElementsAttr::get("
    "::mlir::RankedTensorType::get({}, $_builder.getF" # width # "Type()),"
    "::llvm::ArrayRef($0))";
  let convertFromStorage = "$_self";
}

def F64ElementsAttr : FloatElementsAttr<64>;

// A `width`-bit floating point elements attribute. The attribute should be
// ranked and has a shape as specified in `dims`.
class RankedFloatElementsAttr<int width, list<int> dims> : ElementsAttrBase<
  CPred<"::llvm::isa<::mlir::DenseFPElementsAttr>($_self) &&"
      "::llvm::cast<::mlir::DenseFPElementsAttr>($_self).getType()."
      "getElementType().isF" # width # "() && "
      // Check that this is ranked and has the specified shape.
      "::llvm::cast<::mlir::DenseFPElementsAttr>($_self).getType().hasRank() && "
      "::llvm::cast<::mlir::DenseFPElementsAttr>($_self).getType().getShape() == "
      "::mlir::ArrayRef<int64_t>({" # !interleave(dims, ", ") # "})">,
  width # "-bit float elements attribute of shape [" #
  !interleave(dims, ", ") # "]"> {

  let storageType = [{ ::mlir::DenseFPElementsAttr }];
  let returnType = [{ ::mlir::DenseFPElementsAttr }];

  let constBuilderCall = "::llvm::cast<::mlir::DenseFPElementsAttr>("
  "::mlir::DenseElementsAttr::get("
    "::mlir::RankedTensorType::get({" # !interleave(dims, ", ") #
    "}, $_builder.getF" # width # "Type()), "
    "::llvm::ArrayRef($0)))";
  let convertFromStorage = "$_self";
}

class RankedF32ElementsAttr<list<int> dims> : RankedFloatElementsAttr<32, dims>;
class RankedF64ElementsAttr<list<int> dims> : RankedFloatElementsAttr<64, dims>;

def StringElementsAttr : ElementsAttrBase<
  CPred<"::llvm::isa<::mlir::DenseStringElementsAttr>($_self)" >,
  "string elements attribute"> {

  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

// Attributes containing affine maps.
def AffineMapAttr : Attr<
CPred<"::llvm::isa<::mlir::AffineMapAttr>($_self)">, "AffineMap attribute"> {
  let storageType = [{::mlir::AffineMapAttr }];
  let returnType = [{ ::mlir::AffineMap }];
  let valueType = Index;
  let constBuilderCall = "::mlir::AffineMapAttr::get($0)";
}

// Base class for array attributes.
class ArrayAttrBase<Pred condition, string summary> : Attr<condition, summary> {
  let storageType = [{ ::mlir::ArrayAttr }];
  let returnType = [{ ::mlir::ArrayAttr }];
  let valueType = NoneType;
  let convertFromStorage = "$_self";
  let constBuilderCall = "$_builder.getArrayAttr($0)";
}

def ArrayAttr : ArrayAttrBase<CPred<"::llvm::isa<::mlir::ArrayAttr>($_self)">,
                              "array attribute">;

// Base class for array attributes whose elements are of the same kind.
// `element` specifies the element attribute kind stored in this array.
class TypedArrayAttrBase<Attr element, string summary>: ArrayAttrBase<
    And<[
      // Guarantee this is an ArrayAttr first
      CPred<"::llvm::isa<::mlir::ArrayAttr>($_self)">,
      // Guarantee all elements satisfy the constraints from `element`
      Concat<"::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self), "
                            "[&](::mlir::Attribute attr) { return attr && (",
                               SubstLeaves<"$_self", "attr", element.predicate>,
                            "); })">]>,
    summary> {

  Attr elementAttr = element;
}

def LocationArrayAttr : TypedArrayAttrBase<LocationAttr,
                                           "location array attribute">;

def AffineMapArrayAttr : TypedArrayAttrBase<AffineMapAttr,
                                      "AffineMap array attribute"> {
  let constBuilderCall = "$_builder.getAffineMapArrayAttr($0)";
}

def BoolArrayAttr : TypedArrayAttrBase<BoolAttr,
                                      "1-bit boolean array attribute"> {
  let constBuilderCall = "$_builder.getBoolArrayAttr($0)";
}
def I32ArrayAttr : TypedArrayAttrBase<I32Attr,
                                      "32-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI32ArrayAttr($0)";
}
def I64ArrayAttr : TypedArrayAttrBase<I64Attr,
                                      "64-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI64ArrayAttr($0)";
}
// Variant of I64ArrayAttr whose user accessor is SmallVector<in64_t>.
def I64SmallVectorArrayAttr :
    TypedArrayAttrBase<I64Attr, "64-bit integer array attribute"> {
  let returnType = [{ ::llvm::SmallVector<int64_t, 8> }];
  let convertFromStorage = [{
    llvm::to_vector<4>(
      llvm::map_range($_self.getAsRange<mlir::IntegerAttr>(),
      [](IntegerAttr attr) { return attr.getInt(); }));
  }];
  let constBuilderCall = "$_builder.getI64ArrayAttr($0)";
}
def F32ArrayAttr : TypedArrayAttrBase<F32Attr, "32-bit float array attribute"> {
  let constBuilderCall = "$_builder.getF32ArrayAttr($0)";
}
def F64ArrayAttr : TypedArrayAttrBase<F64Attr, "64-bit float array attribute"> {
  let constBuilderCall = "$_builder.getF64ArrayAttr($0)";
}
def StrArrayAttr : TypedArrayAttrBase<StrAttr, "string array attribute"> {
  let constBuilderCall = "$_builder.getStrArrayAttr($0)";
}
def TypeArrayAttr : TypedArrayAttrBase<TypeAttr, "type array attribute"> {
  let constBuilderCall = "$_builder.getTypeArrayAttr($0)";
}
def IndexListArrayAttr :
  TypedArrayAttrBase<I64ArrayAttr, "Array of 64-bit integer array attributes">;
def DictArrayAttr :
  TypedArrayAttrBase<DictionaryAttr, "Array of dictionary attributes">;

// Attributes containing symbol references.
def SymbolRefAttr : Attr<CPred<"::llvm::isa<::mlir::SymbolRefAttr>($_self)">,
                        "symbol reference attribute"> {
  let storageType = [{ ::mlir::SymbolRefAttr }];
  let returnType = [{ ::mlir::SymbolRefAttr }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self";
}

def FlatSymbolRefAttr : Attr<CPred<"::llvm::isa<::mlir::FlatSymbolRefAttr>($_self)">,
                                   "flat symbol reference attribute"> {
  let storageType = [{ ::mlir::FlatSymbolRefAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getValue()";
}

def SymbolRefArrayAttr :
  TypedArrayAttrBase<SymbolRefAttr, "symbol ref array attribute"> {
  let constBuilderCall = ?;
}

def FlatSymbolRefArrayAttr :
  TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// Derive attribute kinds

// DerivedAttr are attributes whose value is computed from properties
// of the operation. They do not require additional storage and are
// materialized as needed.
// Note: All derived attributes should be materializable as an Attribute. E.g.,
// do not use DerivedAttr for things that could not have been stored as
// Attribute.
//
class DerivedAttr<code ret, code b, code convert = ""> :
    Attr<CPred<"true">, "derived attribute"> {
  let returnType = ret;
  code body = b;

  // Specify how to convert from the derived attribute to an attribute.
  //
  // ## Special placeholders
  //
  // Special placeholders can be used to refer to entities during conversion:
  //
  // * `$_builder` will be replaced by a mlir::Builder instance.
  // * `$_ctxt` will be replaced by the MLIRContext* instance.
  // * `$_self` will be replaced with the derived attribute (value produces
  //    `returnType`).
  let convertFromStorage = convert;
}

// Derived attribute that returns a mlir::Type.
class DerivedTypeAttr<code body> : DerivedAttr<"::mlir::Type", body> {
  let convertFromStorage = "::mlir::TypeAttr::get($_self)";
}

//===----------------------------------------------------------------------===//
// Constant attribute kinds

// Represents a constant attribute of specific Attr type. A constant
// attribute can be specified only of attributes that have a constant
// builder call defined. The constant value is specified as a string.
//
// If used as a constraint, it generates a matcher on a constant attribute by
// using the constant value builder of the attribute and the value.
class ConstantAttr<Attr attribute, string val> : AttrConstraint<
    CPred<"$_self == " # !subst("$0", val, attribute.constBuilderCall)>,
    "constant attribute " # val> {
  Attr attr = attribute;
  string value = val;
}

class ConstF32Attr<string val> : ConstantAttr<F32Attr, val>;
def ConstBoolAttrFalse : ConstantAttr<BoolAttr, "false">;
def ConstBoolAttrTrue : ConstantAttr<BoolAttr, "true">;
def ConstUnitAttr : ConstantAttr<UnitAttr, "true">;

// Constant string-based attribute. Wraps the desired string in escaped quotes.
class ConstantStrAttr<Attr attribute, string val>
    : ConstantAttr<attribute, "\"" # val # "\"">;

//===----------------------------------------------------------------------===//
// Common attribute constraints
//===----------------------------------------------------------------------===//

// A general mechanism to further confine the given `attr` with all the
// `constraints`. This allows to compose complex constraints out of a series
// of more primitive ones.
class ConfinedAttr<Attr attr, list<AttrConstraint> constraints> : Attr<
    And<!listconcat([attr.predicate],
                      !foreach(pred, constraints, pred.predicate))>,
    !foldl(/*init*/attr.summary, /*list*/constraints,
           prev, cur, prev # " " # cur.summary)> {
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = attr.defaultValue;
  let valueType = attr.valueType;
  let isOptional = attr.isOptional;

  let baseAttr = attr;
}

// An AttrConstraint that holds if all attr constraints specified in
// 'constraints' hold.
class AllAttrOf<list<AttrConstraint> constraints> : AttrConstraint<
    And<!listconcat([!head(constraints).predicate],
                    !foreach(pred, !tail(constraints), pred.predicate))>,
    !interleave(!foreach(con, constraints, con.summary), " and ")> {
}

class IntNEQValue<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getInt() != " # n>,
    "whose minimum value is " # n>;

class IntMinValue<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getInt() >= " # n>,
    "whose minimum value is " # n>;

class IntMaxValue<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getInt() <= " # n>,
    "whose maximum value is " # n>;

def IntNonNegative : AttrConstraint<
    CPred<"!::llvm::cast<::mlir::IntegerAttr>($_self).getValue().isNegative()">,
    "whose value is non-negative">;

def IntPositive : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getValue().isStrictlyPositive()">,
    "whose value is positive">;

class ArrayMinCount<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() >= " # n>,
    "with at least " # n # " elements">;

class ArrayCount<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() == " #n>,
    "with exactly " # n # " elements">;

class DenseArrayCount<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::DenseArrayAttr>($_self).size() == " #n>,
    "with exactly " # n # " elements">;

class DenseArrayStrictlyPositive<DenseArrayAttrBase arrayType> : AttrConstraint<
  CPred<"::llvm::all_of(::llvm::cast<" # arrayType #">($_self).asArrayRef(), "
                        "[&](auto v) { return v > 0; })">,
  "whose value is positive">;

class DenseArrayNonNegative<DenseArrayAttrBase arrayType> : AttrConstraint<
  CPred<"::llvm::all_of(::llvm::cast<" # arrayType #">($_self).asArrayRef(), "
                        "[&](auto v) { return v >= 0; })">,
  "whose value is non-negative">;

class DenseArraySorted<DenseArrayAttrBase arrayType> : AttrConstraint<
    CPred<"llvm::is_sorted(::llvm::cast<" # arrayType # ">($_self).asArrayRef())">,
    "should be in non-decreasing order">;

class DenseArrayStrictlySorted<DenseArrayAttrBase arrayType> : AttrConstraint<
    And<[
      CPred<"llvm::is_sorted(::llvm::cast<" # arrayType # ">($_self).asArrayRef())">,
      // Check that no two adjacent elements are the same.
      CPred<"[](" # arrayType.returnType # " a) {\n"
        "return std::adjacent_find(std::begin(a), std::end(a)) == "
        "std::end(a);\n"
        "}(::llvm::cast<" # arrayType # ">($_self).asArrayRef())"
      >]>,
    "should be in increasing order">;

class IntArrayNthElemEq<int index, int value> : AttrConstraint<
    And<[
      CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() > " # index>,
      CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<::mlir::ArrayAttr>($_self)["
            # index # "]).getInt() == "  # value>
       ]>,
    "whose " # index # "-th element must be " # value>;

class IntArrayNthElemMinValue<int index, int min> : AttrConstraint<
    And<[
      CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() > " # index>,
      CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<::mlir::ArrayAttr>($_self)["
            # index # "]).getInt() >= " # min>
        ]>,
    "whose " # index # "-th element must be at least " # min>;

class IntArrayNthElemMaxValue<int index, int max> : AttrConstraint<
    And<[
      CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() > " # index>,
      CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<::mlir::ArrayAttr>($_self)["
            # index # "]).getInt() <= " # max>
        ]>,
    "whose " # index # "-th element must be at most " # max>;

class IntArrayNthElemInRange<int index, int min, int max> : AttrConstraint<
    And<[
      CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() > " # index>,
      CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<::mlir::ArrayAttr>($_self)["
            # index # "]).getInt() >= " # min>,
      CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<::mlir::ArrayAttr>($_self)["
            # index # "]).getInt() <= " # max>
        ]>,
    "whose " # index # "-th element must be at least " # min # " and at most " # max>;

def IsNullAttr : AttrConstraint<
    CPred<"!$_self">, "empty attribute (for optional attributes)">;

//===----------------------------------------------------------------------===//
// Region definitions
//===----------------------------------------------------------------------===//

class Region<Pred condition, string descr = ""> :
    RegionConstraint<condition, descr>;

// Any region.
def AnyRegion : Region<CPred<"true">, "any region">;

// A region with the given number of blocks.
class SizedRegion<int numBlocks> : Region<
  CPred<"::llvm::hasNItems($_self, " # numBlocks # ")">,
  "region with " # numBlocks # " blocks">;

// A region with at least the given number of blocks.
class MinSizedRegion<int numBlocks> : Region<
  CPred<"::llvm::hasNItemsOrMore($_self, " # numBlocks # ")">,
  "region with at least " # numBlocks # " blocks">;

// A region with at most the given number of blocks.
class MaxSizedRegion<int numBlocks> : Region<
  CPred<"::llvm::hasNItemsOrLess($_self, " # numBlocks # ")">,
  "region with at most " # numBlocks # " blocks">;

// A variadic region constraint. It expands to zero or more of the base region.
class VariadicRegion<Region region>
  : Region<region.predicate, region.summary>;


//-------------------------------------------------------------------------===//
// AttrTrait definitions
//===----------------------------------------------------------------------===//

// These classes are used to define attribute specific traits.

// Specify attribute specific declarations and definitions in `extraAttrDeclaration`
// and `extraAttrDefinition` template arguments.
class NativeAttrTrait<string name,
                      code extraAttrDeclaration = [{}],
                      code extraAttrDefinition = [{}]>
    : NativeTrait<name, "Attribute", extraAttrDeclaration, extraAttrDefinition>;

class ParamNativeAttrTrait<string prop, string params>
    : ParamNativeTrait<prop, params, "Attribute">;
class GenInternalAttrTrait<string prop> : GenInternalTrait<prop, "Attribute">;
class PredAttrTrait<string descr, Pred pred> : PredTrait<descr, pred>;

//===----------------------------------------------------------------------===//
// TypeTrait definitions
//===----------------------------------------------------------------------===//

// These classes are used to define type specific traits.

// Specify type specific declarations and definitions in `extraTypeDeclaration`
// and `extraTypeDefinition` template arguments.
class NativeTypeTrait<string name,
                      code extraTypeDeclaration = [{}],
                      code extraTypeDefinition = [{}]>
    : NativeTrait<name, "Type", extraTypeDeclaration, extraTypeDefinition>;

class ParamNativeTypeTrait<string prop, string params>
    : ParamNativeTrait<prop, params, "Type">;
class GenInternalTypeTrait<string prop> : GenInternalTrait<prop, "Type">;
class PredTypeTrait<string descr, Pred pred> : PredTrait<descr, pred>;

//===----------------------------------------------------------------------===//
// Builders
//===----------------------------------------------------------------------===//

// Class for defining a custom getter.
//
// TableGen generates several generic getter methods for each attribute and type
// by default, corresponding to the specified dag parameters. If the default
// generated ones cannot cover some use case, custom getters can be defined
// using instances of this class.
//
// The signature of the `get` is always either:
//
// ```c++
// static <ClassName> get(MLIRContext *context, <other-parameters>...) {
//   <body>...
// }
// ```
//
// or:
//
// ```c++
// static <ClassName> get(MLIRContext *context, <parameters>...);
// ```
//
// To define a custom getter, the parameter list and body should be passed
// in as separate template arguments to this class. The parameter list is a
// TableGen DAG with `ins` operation with named arguments, which has either:
//   - string initializers ("Type":$name) to represent a typed parameter, or
//   - CArg-typed initializers (CArg<"Type", "default">:$name) to represent a
//     typed parameter that may have a default value.
// The type string is used verbatim to produce code and, therefore, must be a
// valid C++ type. It is used inside the C++ namespace of the parent Type's
// dialect; explicit namespace qualification like `::mlir` may be necessary if
// Types are not placed inside the `mlir` namespace. The default value string is
// used verbatim to produce code and must be a valid C++ initializer the given
// type. For example, the following signature specification
//
// ```
// AttrOrTypeBuilder<(ins "int":$integerArg, CArg<"float", "3.0f">:$floatArg)>
// ```
//
// has an integer parameter and a float parameter with a default value.
//
// If an empty string is passed in for `body`, then *only* the builder
// declaration will be generated; this provides a way to define complicated
// builders entirely in C++. If a `body` string is provided, the `Base::get`
// method should be invoked using `$_get`, e.g.:
//
// ```
// AttrOrTypeBuilder<(ins "int":$integerArg, CArg<"float", "3.0f">:$floatArg), [{
//   return $_get($_ctxt, integerArg, floatArg);
// }]>
// ```
//
// This is necessary because the `body` is also used to generate `getChecked`
// methods, which have a different underlying `Base::get*` call.
//
class AttrOrTypeBuilder<dag parameters, code bodyCode = "",
                        string returnTypeStr = ""> {
  dag dagParams = parameters;
  code body = bodyCode;

  // Change the return type of the builder. By default, it is the type of the
  // attribute or type.
  string returnType = returnTypeStr;

  // The context parameter can be inferred from one of the other parameters and
  // is not implicitly added to the parameter list.
  bit hasInferredContextParam = 0;
}
class AttrBuilder<dag parameters, code bodyCode = "", string returnType = "">
  : AttrOrTypeBuilder<parameters, bodyCode, returnType>;
class TypeBuilder<dag parameters, code bodyCode = "", string returnType = "">
  : AttrOrTypeBuilder<parameters, bodyCode, returnType>;

// A class of AttrOrTypeBuilder that is able to infer the MLIRContext parameter
// from one of the other builder parameters. Instances of this builder do not
// have `MLIRContext *` implicitly added to the parameter list.
class AttrOrTypeBuilderWithInferredContext<dag parameters, code bodyCode = "",
                                           string returnType = "">
  : TypeBuilder<parameters, bodyCode, returnType> {
  let hasInferredContextParam = 1;
}
class AttrBuilderWithInferredContext<dag parameters, code bodyCode = "",
                                     string returnType = "">
  : AttrOrTypeBuilderWithInferredContext<parameters, bodyCode, returnType>;
class TypeBuilderWithInferredContext<dag parameters, code bodyCode = "",
                                     string returnType = "">
  : AttrOrTypeBuilderWithInferredContext<parameters, bodyCode, returnType>;

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

// Define a new attribute or type, named `name`, that inherits from the given
// C++ base class.
class AttrOrTypeDef<string valueType, string name, list<Trait> defTraits,
                    string baseCppClass> {
  // The name of the C++ base class to use for this def.
  string cppBaseClassName = baseCppClass;

  // Additional, longer human-readable description of what the def does.
  string description = "";

  // Name of storage class to generate or use.
  string storageClass = name # valueType # "Storage";

  // Namespace (withing dialect c++ namespace) in which the storage class
  // resides.
  string storageNamespace = "detail";

  // Specify if the storage class is to be generated.
  bit genStorageClass = 1;

  // Specify that the generated storage class has a constructor which is written
  // in C++.
  bit hasStorageCustomConstructor = 0;

  // The list of parameters for this type. Parameters will become both
  // parameters to the get() method and storage class member variables.
  //
  // The format of this dag is:
  //    (ins
  //        "<c++ type>":$param1Name,
  //        "<c++ type>":$param2Name,
  //        AttrOrTypeParameter<"c++ type", "param description">:$param3Name)
  // AttrOrTypeParameters (or more likely one of their subclasses) are required
  // to add more information about the parameter, specifically:
  //  - Documentation
  //  - Code to allocate the parameter (if allocation is needed in the storage
  //    class constructor)
  //
  // For example:
  //    (ins "int":$width,
  //         ArrayRefParameter<"bool", "list of bools">:$yesNoArray)
  //
  // (ArrayRefParameter is a subclass of AttrOrTypeParameter which has
  // allocation code for re-allocating ArrayRefs. It is defined below.)
  dag parameters = (ins);

  // Custom builder methods.
  // In addition to the custom builders provided here, and unless
  // skipDefaultBuilders is set, a default builder is generated with the
  // following signature:
  //
  // ```c++
  // static <ClassName> get(MLIRContext *, <parameters>);
  // ```
  //
  // Note that builders should only be provided when a def has parameters.
  list<AttrOrTypeBuilder> builders = ?;

  // The list of traits attached to this def.
  list<Trait> traits = defTraits;

  // Use the lowercased name as the keyword for parsing/printing. Specify only
  // if you want tblgen to generate declarations and/or definitions of
  // the printer/parser. If specified and the Attribute or Type contains
  // parameters, `assemblyFormat` or `hasCustomAssemblyFormat` must also be
  // specified.
  string mnemonic = ?;

  // Custom assembly format. Requires 'mnemonic' to be specified. Cannot be
  // specified at the same time as 'hasCustomAssemblyFormat'. The generated
  // printer requires 'genAccessors' to be true.
  string assemblyFormat = ?;
  /// This field indicates that the attribute or type has a custom assembly format
  /// implemented in C++. When set to `1` a `parse` and `print` method are generated
  /// on the generated class. The attribute or type should implement these methods to
  /// support the custom format.
  bit hasCustomAssemblyFormat = 0;

  // If set, generate accessors for each parameter.
  bit genAccessors = 1;

  // Avoid generating default get/getChecked functions. Custom get methods must
  // be provided.
  bit skipDefaultBuilders = 0;

  // Generate the verify and getChecked methods.
  bit genVerifyDecl = 0;

  // Extra code to include in the class declaration.
  code extraClassDeclaration = [{}];

  // Additional code that will be added to the generated source file. The
  // generated code is placed inside the class's C++ namespace. `$cppClass` is
  // replaced by the class name.
  code extraClassDefinition = [{}];
}

// Define a new attribute, named `name`, belonging to `dialect` that inherits
// from the given C++ base class.
class AttrDef<Dialect dialect, string name, list<Trait> traits = [],
              string baseCppClass = "::mlir::Attribute">
    : DialectAttr<dialect, CPred<"">, /*descr*/"">,
      AttrOrTypeDef<"Attr", name, traits, baseCppClass> {
  // The name of the C++ Attribute class.
  string cppClassName = name # "Attr";
  let storageType = dialect.cppNamespace # "::" # name # "Attr";

  // The underlying C++ value type
  let returnType = dialect.cppNamespace # "::" # cppClassName;

  // Make it possible to use such attributes as parameters for other attributes.
  string cppType = dialect.cppNamespace # "::" # cppClassName;

  // The call expression to convert from the storage type to the return
  // type. For example, an enum can be stored as an int but returned as an
  // enum class.
  //
  // Format: $_self will be expanded to the attribute.
  //
  // For example, `$_self.getValue().getSExtValue()` for `IntegerAttr val` will
  // expand to `getAttrOfType<IntegerAttr>("val").getValue().getSExtValue()`.
  let convertFromStorage = "::llvm::cast<" # dialect.cppNamespace #
                                 "::" # cppClassName # ">($_self)";

  // The predicate for when this def is used as a constraint.
  let predicate = CPred<"::llvm::isa<" # dialect.cppNamespace #
                                 "::" # cppClassName # ">($_self)">;
}

// Define a new type, named `name`, belonging to `dialect` that inherits from
// the given C++ base class.
class TypeDef<Dialect dialect, string name, list<Trait> traits = [],
              string baseCppClass = "::mlir::Type">
    : DialectType<dialect, CPred<"">, /*descr*/"", name # "Type">,
      AttrOrTypeDef<"Type", name, traits, baseCppClass> {
  // Make it possible to use such type as parameters for other types.
  string cppType = dialect.cppNamespace # "::" # cppClassName;

  // A constant builder provided when the type has no parameters.
  let builderCall = !if(!empty(parameters),
                           "$_builder.getType<" # dialect.cppNamespace #
                               "::" # cppClassName # ">()",
                           "");
  // The predicate for when this def is used as a constraint.
  let predicate = CPred<"::llvm::isa<" # dialect.cppNamespace #
                                 "::" # cppClassName # ">($_self)">;
}

//===----------------------------------------------------------------------===//
// Parameters
//===----------------------------------------------------------------------===//

// 'Parameters' should be subclasses of this or simple strings (which is a
// shorthand for AttrOrTypeParameter<"C++Type">).
class AttrOrTypeParameter<string type, string desc, string accessorType = ""> {
  // Custom memory allocation code for storage constructor.
  code allocator = ?;
  // Comparator used to compare two instances for equality. By default, it uses
  // the C++ equality operator.
  code comparator = ?;
  // The C++ type of this parameter.
  string cppType = type;
  // The C++ type of the accessor for this parameter.
  string cppAccessorType = !if(!empty(accessorType), type, accessorType);
  // The C++ storage type of of this parameter if it is a reference, e.g.
  // `std::string` for `StringRef` or `SmallVector` for `ArrayRef`.
  string cppStorageType = cppType;
  // The C++ code to convert from the storage type to the parameter type.
  string convertFromStorage = "$_self";
  // One-line human-readable description of the argument.
  string summary = desc;
  // The format string for the asm syntax (documentation only).
  string syntax = ?;
  // The default parameter parser is `::mlir::FieldParser<T>::parse($_parser)`,
  // which returns `FailureOr<T>`. Specialize `FieldParser` to support parsing
  // for your type. Or you can provide a customer printer. For attributes,
  // "$_type" will be replaced with the required attribute type.
  string parser = ?;
  // The default parameter printer is `$_printer << $_self`. Overload the stream
  // operator of `AsmPrinter` as necessary to print your type. Or you can
  // provide a custom printer.
  string printer = ?;
  // Provide a default value for the parameter. Parameters with default values
  // are considered optional. If a value was not parsed for the parameter, it
  // will be set to the default value. Parameters equal to their default values
  // are elided when printing. Equality is checked using the `comparator` field,
  // which by default is the C++ equality operator. The current MLIR context is
  // made available through `$_ctxt`, e.g., for constructing default values for
  // attributes and types.
  string defaultValue = "";
}
class AttrParameter<string type, string desc, string accessorType = "">
 : AttrOrTypeParameter<type, desc, accessorType>;
class TypeParameter<string type, string desc, string accessorType = "">
 : AttrOrTypeParameter<type, desc, accessorType>;

// An optional parameter.
class OptionalParameter<string type, string desc = ""> :
    AttrOrTypeParameter<type, desc> {
  let defaultValue = cppStorageType # "()";
}

// A parameter with a default value.
class DefaultValuedParameter<string type, string value, string desc = ""> :
    AttrOrTypeParameter<type, desc> {
  let defaultValue = value;
}

// For StringRefs, which require allocation.
class StringRefParameter<string desc = "", string value = ""> :
    AttrOrTypeParameter<"::llvm::StringRef", desc> {
  let allocator = [{$_dst = $_allocator.copyInto($_self);}];
  let printer = [{$_printer << '"' << $_self << '"';}];
  let cppStorageType = "std::string";
  let defaultValue = value;
}

// For APFloats, which require comparison.
class APFloatParameter<string desc> :
    AttrOrTypeParameter<"::llvm::APFloat", desc> {
  let comparator = "$_lhs.bitwiseIsEqual($_rhs)";
}

// For standard ArrayRefs, which require allocation.
class ArrayRefParameter<string arrayOf, string desc = ""> :
    AttrOrTypeParameter<"::llvm::ArrayRef<" # arrayOf # ">", desc> {
  let allocator = [{$_dst = $_allocator.copyInto($_self);}];
  let cppStorageType = "::llvm::SmallVector<" # arrayOf # ">";
}

// Regular array parameters cannot be parsed when empty. This optional array
// parameter can be used with optional groups to be parsed when empty.
class OptionalArrayRefParameter<string arrayOf, string desc = ""> :
    OptionalParameter<"::llvm::ArrayRef<" # arrayOf # ">", desc> {
  let allocator = [{$_dst = $_allocator.copyInto($_self);}];
  let cppStorageType = "::llvm::SmallVector<" # arrayOf # ">";
  let comparator = cppType # "($_lhs) == " # cppType # "($_rhs)";
}

// For classes which require allocation and have their own allocateInto method.
class SelfAllocationParameter<string type, string desc> :
    AttrOrTypeParameter<type, desc> {
  let allocator = [{$_dst = $_self.allocateInto($_allocator);}];
}

// For ArrayRefs which contain things which allocate themselves.
class ArrayRefOfSelfAllocationParameter<string arrayOf, string desc> :
    AttrOrTypeParameter<"::llvm::ArrayRef<" # arrayOf # ">", desc> {
  let allocator = [{
    llvm::SmallVector<}] # arrayOf # [{, 4> tmpFields;
    for (size_t i = 0, e = $_self.size(); i < e; ++i)
      tmpFields.push_back($_self[i].allocateInto($_allocator));
    $_dst = $_allocator.copyInto(ArrayRef<}] # arrayOf # [{>(tmpFields));
  }];
}

// This is a special attribute parameter that represents the "self" type of the
// attribute. It is specially handled by the assembly format generator to derive
// its value from the optional trailing type after each attribute.
//
// By default, the self type parameter is optional and has a default value of
// `none`. If a derived type other than `::mlir::Type` is specified, the
// parameter loses its default value unless another one is specified by
// `typeBuilder`.
class AttributeSelfTypeParameter<string desc,
                                 string derivedType = "::mlir::Type",
                                 string typeBuilder = ""> :
    AttrOrTypeParameter<derivedType, desc> {
  let defaultValue = !if(!and(!empty(typeBuilder),
                              !eq(derivedType, "::mlir::Type")),
                         "::mlir::NoneType::get($_ctxt)", typeBuilder);
}

//===----------------------------------------------------------------------===//
// ArrayOfAttr
//===----------------------------------------------------------------------===//

/// This class defines an attribute that contains an array of elements. The
/// elements can be any type, but if they are attributes, the nested elements
/// are parsed and printed using the custom attribute syntax.
class ArrayOfAttr<Dialect dialect, string attrName, string attrMnemonic,
                  string eltName, list<Trait> traits = []>
    : AttrDef<dialect, attrName, traits> {
  let parameters = (ins OptionalArrayRefParameter<eltName>:$value);
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`[` (`]`) : ($value^ `]`)?";

  let returnType = "::llvm::ArrayRef<" # eltName # ">";
  let constBuilderCall = "$_builder.getAttr<" # attrName # "Attr>($0)";
  let convertFromStorage = "$_self.getValue()";

  let extraClassDeclaration = [{
    auto begin() const { return getValue().begin(); }
    auto end() const { return getValue().end(); }
    bool empty() const { return getValue().empty(); }
    size_t size() const { return getValue().size(); }
    auto &front() const { return getValue().front(); }
    auto &back() const { return getValue().back(); }
    auto &operator[](size_t index) { return getValue()[index]; }
    operator }] # returnType # [{() const { return getValue(); }
  }];
}

#endif // ATTRTYPEBASE_TD
