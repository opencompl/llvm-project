//===-- Traits.td - Trait definations file ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definations for traits.
//
//===----------------------------------------------------------------------===//

#ifndef TRAITS_TD
#define TRAITS_TD

include "mlir/IR/Constraints.td"

//===----------------------------------------------------------------------===//
// Trait definitions
//===----------------------------------------------------------------------===//

// Trait represents a trait regarding an attribute, operation, or type.
class Trait;

// Define a Trait corresponding to a list of Traits, this allows for specifying
// a list of traits as trait. Avoids needing to do `[Traits, ...] # ListOfTraits
// # [Others, ...]` while still allowing providing convenient groupings.
class TraitList<list<Trait> props> : Trait {
  list<Trait> traits = props;
}

// NativeTrait corresponds to the MLIR C++ trait mechanism. The purpose to wrap
// around C++ symbol string with this class is to make traits specified for
// entities in TableGen less alien and more integrated.
// `extraConcreteClassDeclaration` and `extraConcreteClassDefinition` code
// get injected into the entities in which the NativeTrait is specified for.
class NativeTrait<string name, string entityType,
                    code extraClassDeclaration = [{}],
                    code extraClassDefinition = [{}]> : Trait {
  string trait = name;
  string cppNamespace = "::mlir::" # entityType # "Trait";

  code extraConcreteClassDeclaration = extraClassDeclaration;
  code extraConcreteClassDefinition = extraClassDefinition;
}

// ParamNativeTrait corresponds to the template-parameterized traits in the C++
// implementation. MLIR uses nested class templates to implement such traits
// leading to constructs of the form "TraitName<Parameters>::Impl". Use the
// value in `prop` as the trait name and the value in `params` as parameters to
// construct the native trait class name.
class ParamNativeTrait<string prop, string params, string entityType>
    : NativeTrait<prop # "<" # params # ">::Impl", entityType>;

// GenInternalTrait is a trait that does not have direct C++ mapping but affects
// an entities definition generator internals, like how operation builders and
// operand/attribute/result getters are generated.
class GenInternalTrait<string prop, string entityType> : Trait {
  string trait = "::mlir::" # entityType # "Trait::" # prop;
}

// PredTrait is a trait implemented by way of a predicate on an entity.
class PredTrait<string descr, Pred pred> : Trait {
  string summary = descr;
  Pred predicate = pred;
}

//===----------------------------------------------------------------------===//
// OpTrait definitions
//===----------------------------------------------------------------------===//

// A trait that describes the structure of operation will be marked with
// `StructuralOpTrait` and they will be verified first.
class StructuralOpTrait;

// These classes are used to define operation specific traits.

// Specify op specific declarations and definitions in `extraOpDeclaration`
// and `extraOpDefinition` template arguments.
class NativeOpTrait<string name, list<Trait> traits = [],
                    code extraOpDeclaration = [{}],
                    code extraOpDefinition = [{}]>
    : NativeTrait<name, "Op", extraOpDeclaration, extraOpDefinition> {
  // Specify the list of traits that need to be verified before the verification
  // of this NativeOpTrait.
  list<Trait> dependentTraits = traits;
}
class ParamNativeOpTrait<string prop, string params,
                         list<Trait> traits = []>
    : ParamNativeTrait<prop, params, "Op"> {
  // Specify the list of traits that need to be verified before the verification
  // of this ParamNativeOpTrait.
  list<Trait> dependentTraits = traits;
}
class GenInternalOpTrait<string prop, list<Trait> traits = []>
    : GenInternalTrait<prop, "Op"> {
  // Specify the list of traits that need to be verified before the verification
  // of this GenInternalOpTrait.
  list<Trait> dependentTraits = traits;
}
class PredOpTrait<string descr, Pred pred, list<Trait> traits = []>
    : PredTrait<descr, pred> {
  // Specify the list of traits that need to be verified before the verification
  // of this PredOpTrait.
  list<Trait> dependentTraits = traits;
}

// Op defines an affine scope.
def AffineScope : NativeOpTrait<"AffineScope">;
// Op defines an automatic allocation scope.
def AutomaticAllocationScope :
  NativeOpTrait<"AutomaticAllocationScope">;
// Op supports operand broadcast behavior.
def ResultsBroadcastableShape :
  NativeOpTrait<"ResultsBroadcastableShape">;
// X op Y == Y op X
def Commutative  : NativeOpTrait<"IsCommutative">;
// op op X == op X (unary) / X op X == X (binary)
// FIXME: Idempotent should depend on SameOperandsAndResultType
def Idempotent : NativeOpTrait<"IsIdempotent">;
// op op X == X
// FIXME: Involution should depend on SameOperandsAndResultType
def Involution : NativeOpTrait<"IsInvolution">;
// Op behaves like a constant.
def ConstantLike : NativeOpTrait<"ConstantLike">;
// Op is isolated from above.
def IsolatedFromAbove : NativeOpTrait<"IsIsolatedFromAbove">;
// Op results are float or vectors/tensors thereof.
def ResultsAreFloatLike : NativeOpTrait<"ResultsAreFloatLike">;
// Op has the same operand type.
def SameTypeOperands : NativeOpTrait<"SameTypeOperands">;
// Op has same shape for all operands.
def SameOperandsShape : NativeOpTrait<"SameOperandsShape">;
// Op has same operand and result shape.
def SameOperandsAndResultShape :
  NativeOpTrait<"SameOperandsAndResultShape">;
// Op has the same element type (or type itself, if scalar) for all operands.
def SameOperandsElementType :
  NativeOpTrait<"SameOperandsElementType">;
// Op has the same operand and result element type (or type itself, if scalar).
def SameOperandsAndResultElementType :
  NativeOpTrait<"SameOperandsAndResultElementType">;
// Op is a terminator.
def Terminator : NativeOpTrait<"IsTerminator">;
// Op can be safely normalized in the presence of MemRefs with
// non-identity maps.
def MemRefsNormalizable : NativeOpTrait<"MemRefsNormalizable">;
// Op is elementwise on tensor/vector operands and results.
def Elementwise : NativeOpTrait<"Elementwise">;
// Elementwise op can be applied to scalars instead tensor/vector operands.
def Scalarizable : NativeOpTrait<"Scalarizable", [Elementwise]>;
// Elementwise op can be applied to all-vector operands.
def Vectorizable : NativeOpTrait<"Vectorizable", [Elementwise]>;
// Elementwise op can be applied to all-tensor operands.
def Tensorizable : NativeOpTrait<"Tensorizable", [Elementwise]>;

// Group together `Elementwise`, `Scalarizable`, `Vectorizable`, and
// `Tensorizable` for convenience.
def ElementwiseMappable : TraitList<[
    Elementwise,
    Scalarizable,
    Vectorizable,
    Tensorizable,
]>;

// Op's regions have a single block.
def SingleBlock : NativeOpTrait<"SingleBlock">, StructuralOpTrait;

// Op's regions have a single block with the specified terminator.
class SingleBlockImplicitTerminator<string op>
    : ParamNativeOpTrait<"SingleBlockImplicitTerminator", op>,
      StructuralOpTrait;

// Op's regions don't have terminator.
def NoTerminator : NativeOpTrait<"NoTerminator">, StructuralOpTrait;

// Op's parent operation is the provided one.
class HasParent<string op>
    : ParamNativeOpTrait<"HasParent", op>, StructuralOpTrait;

class ParentOneOf<list<string> ops>
    : ParamNativeOpTrait<"HasParent", !interleave(ops, ", ")>,
      StructuralOpTrait;

// Op result type is derived from the first attribute. If the attribute is an
// subclass of `TypeAttrBase`, its value is used, otherwise, the type of the
// attribute content is used.
def FirstAttrDerivedResultType :
  GenInternalOpTrait<"FirstAttrDerivedResultType">;

// TODO: Turn the following into normal traits and generate verification for
// them.

// All variadic operands of the op have the same number of values.
// A variadic operand contains an array of values whose array size is only
// known at runtime. This trait requires all variadic operands of an op
// to have the same array size.
def SameVariadicOperandSize : GenInternalOpTrait<"SameVariadicOperandSize">;
// All variadic results of the op have the same number of values.
// A variadic result contains an array of values whose array size is only
// known at runtime. This trait requires all variadic results of an op
// to have the same array size.
def SameVariadicResultSize : GenInternalOpTrait<"SameVariadicResultSize">;

// Uses an attribute named `operand_segment_sizes` to specify how many actual
// operand each ODS-declared operand (variadic or not) corresponds to.
// This trait is used for ops that have multiple variadic operands but do
// not know statically their size relationship. The attribute must be a 1D
// vector that has the same number of elements as the number of ODS declared
// operands. That means even if some operands are non-variadic, the attribute
// still need to have an element for its size, which is always 1.
def AttrSizedOperandSegments :
  NativeOpTrait<"AttrSizedOperandSegments">, StructuralOpTrait;
// Similar to AttrSizedOperandSegments, but used for results. The attribute
// should be named as `result_segment_sizes`.
def AttrSizedResultSegments  :
  NativeOpTrait<"AttrSizedResultSegments">, StructuralOpTrait;

// Op attached regions have no arguments
def NoRegionArguments : NativeOpTrait<"NoRegionArguments">, StructuralOpTrait;

#endif // TRAITS_TD
