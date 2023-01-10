//===- IRDLOps.td - IR Definition Language Dialect ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRDL dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDLOPS
#define MLIR_DIALECT_IRDL_IR_IRDLOPS

include "IRDL.td"
include "IRDLTypes.td"
include "IRDLInterfaces.td"
include "IRDLAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IRDL_Op<string mnemonic, list<Trait> traits = []>
    : Op<IRDL_Dialect, mnemonic, traits> {
}

class AtMostOneChildOf<string op> : ParamNativeOpTrait<"AtMostOneChildOf", op>;

//===----------------------------------------------------------------------===//
// IRDL Dialect definition
//===----------------------------------------------------------------------===//

def IRDL_DialectOp : IRDL_Op<"dialect", [IsolatedFromAbove, NoTerminator]> {
  let summary = "Define a new dialect";
  let description = [{
    `irdl.dialect` operations defines a dialect.

    Example:

    ```mlir
    irdl.dialect cmath {
      ...
    }
    ```

    The above program defines a `cmath` dialect.
  }];

  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "custom<KeywordOrString>($name) attr-dict-with-keyword custom<SingleBlockRegion>($body)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IRDL Type and Attribute parameters definition
//===----------------------------------------------------------------------===//

def IRDL_ParametersOp : IRDL_Op<"parameters", [HasParent<"TypeOp">]> {
  let summary = "Define the parameters of a type/attribute definition";
  let description = [{
    `irdl.parameters` defines new parameters belonging to the type or attribute
    defined by the parent operation.

    Example:

    ```mlir
    irdl.dialect cmath {
      irdl.type complex {
        %0 = irdl.is_type : i32
        %1 = irdl.is_type : i64
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(%2)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath`. The
    type has a single parameter that should be either `i32` or `i64`.
  }];

  let arguments = (ins Variadic<IRDL_ConstraintType>:$args);
  let assemblyFormat = [{ `(` $args `)` attr-dict }];
}

//===----------------------------------------------------------------------===//
// IRDL Type definition
//===----------------------------------------------------------------------===//

def IRDL_TypeOp : IRDL_Op<"type", [HasParent<"DialectOp">, NoTerminator, NoRegionArguments, AtMostOneChildOf<"ParametersOp">]> {
  let summary = "Define a new type";
  let description = [{
    `irdl.type` defines new types belonging to the previously defined dialect
    using `irdl.dialect`.

    The type parameters can be defined with an `irdl.parameters` operation in
    the optional region.

    Example:

    ```mlir
    irdl.dialect cmath {
      irdl.type complex {
        %0 = irdl.is_type : i32
        %1 = irdl.is_type : i64
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(%2)
      }
    }
    ```
    
    The above program defines a type `complex` inside the dialect `cmath`. The
    type has a single parameter that should be either `i32` or `i64`.
  }];

  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "custom<KeywordOrString>($name) attr-dict-with-keyword custom<SingleBlockRegion>($body)";  

  let extraClassDeclaration = [{
    /// Get the parent dialect operation.
    DialectOp getDialectOp() { return cast<DialectOp>(getOperation()->getParentOp()); };
  }];
}

//===----------------------------------------------------------------------===//
// IRDL Operation definition
//===----------------------------------------------------------------------===//

def IRDL_OperandsOp : IRDL_Op<"operands", [HasParent<"OperationOp">]> {
  let summary = "Define the operands of an operation";
  let description = [{
    `irdl.operands` define the operands of the `irdl.operation` parent operation
    definition.

    Example:

    ```mlir
    irdl.dialect testd {
      irdl.operation test_op {
        %0 = irdl.is_type : i32
        %1 = irdl.is_type : i64
        %2 = irdl.any_of(%0, %1)
        irdl.operands(%0, %2)
      }
    }
    ```

    The above program defines an operation `test_op` with two expected operands.
    The first operand is expected to be of type `i32`, and the second operand is
    expected to be either of type `i32` or `i64`.
  }];

  let arguments = (ins Variadic<IRDL_ConstraintType>:$args);
  let assemblyFormat = [{ `(` $args `)` attr-dict }];
}

def IRDL_ResultsOp : IRDL_Op<"results", [HasParent<"OperationOp">]> {
  let summary = "Define the results of an operation";
  let description = [{
    `irdl.results` define the results of the `irdl.operation` parent operation
    definition.

    Example:

    ```mlir
    irdl.dialect testd {
      irdl.operation test_op {
        %0 = irdl.is_type : i32
        %1 = irdl.is_type : i64
        %2 = irdl.any_of(%0, %1)
        irdl.results(%0, %2)
      }
    }
    ```

    The above program defines an operation `test_op` with two expected results.
    The first result is expected to be an `i32`, and the second result is
    expected to be either an `i32` or an `i64`.
  }];

  let arguments = (ins Variadic<IRDL_ConstraintType>:$args);
  let assemblyFormat = [{ `(` $args `)` attr-dict }];
}

def IRDL_OperationOp : IRDL_Op<"operation",
    [HasParent<"DialectOp">, NoTerminator, NoRegionArguments,
    AtMostOneChildOf<"OperandsOp, ResultsOp">]> {
  let summary = "Define a new operation";
  let description = [{
    `irdl.operation` defines new operations belonging to the previously
    defined dialect using `irdl.dialect`.

    Those operations have no verifiers.

    Example:

    ```mlir
    irdl.dialect "cmath" {
      irdl.operation "mul"
    }
    ```
    
    The above program defines an operation `mul` inside the dialect `cmath`.
  }];

  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "custom<KeywordOrString>($name) attr-dict-with-keyword custom<SingleBlockRegion>($body)";

  let extraClassDeclaration = [{
    /// Get the parent dialect operation.
    DialectOp getDialectOp() { return cast<DialectOp>(getOperation()->getParentOp()); };
  }];
}

//===----------------------------------------------------------------------===//
// IRDL Constraint operations
//===----------------------------------------------------------------------===//

class IRDL_ConstraintOp<string mnemonic, list<Trait> traits = []>
    : IRDL_Op<mnemonic, [VerifyConstraintInterface, DeclareOpInterfaceMethods<VerifyConstraintInterface>] # traits> {
}

def IRDL_IsType : IRDL_ConstraintOp<"is_type", [ParentOneOf<["TypeOp", "OperationOp"]>, Pure]> {
  let summary = "Constraints to a specific type";
  let description = [{
    `irdl.is_type` defines a constraint that only accepts a specific type.

    Example:

    ```mlir
    irdl.dialect cmath {
      irdl.type complex_restrictive {
        %0 = irdl.is_type : i32
        irdl.parameters(%0)
      }
    }
    ```

    The above program defines a type `complex_restrictive` inside the dialect `cmath` that
    can only have `i32` as its parameter.

    `irdl.is_type` also accepts completely instantiated parametric types,
    such as `std.complex<f32>`. It does not however allow expressing asbtract
    types, such as `std.complex<T>` where `T` is a type constraint. See
    `irdl.parametric_type` instead for this use case.

    Example:

    ```mlir
    irdl.dialect cmath {
      irdl.operation extract_f32 {
        %0 = irdl.is_type : "std.complex"<f32>
        %1 = irdl.is_type : f32
        irdl.parameters(%0)
        irdl.results(%1)
      }
    }
    ```

    The above programe defines an operation `extract_f32` inside the dialect `cmath` that
    takes as parameter exactly `std.complex<f32>` and returns an `f32`.
  }];

  let arguments = (ins ParamTypeAttrOrAnyAttr:$expected);
  let results = (outs IRDL_ConstraintType:$output);
  let assemblyFormat = [{ `:` $expected attr-dict }];
}

def IRDL_ParametricType : IRDL_ConstraintOp<"parametric_type", [ParentOneOf<["TypeOp", "OperationOp"]>, Pure]> {
  let summary = "Constraints to a specific instantiated type";
  let description = [{
    `irdl.parametric_type` defines a constraint that accepts either a type defined
    in IRDL or a type wrapper. If the requested type is parametric, it will
    additionally take as operands constraints on its type arguments.

    Example:

    ```mlir
    irdl.dialect cmath {
      // ...

      irdl.operation norm {
        %0 = irdl.any_type
        %1 = irdl.parametric_type : "cmath.complex"<%0>
        irdl.operands(%1)
        irdl.results(%0)
      }
    }
    ```

    The above program defines an operation `norm` inside the dialect `cmath` that
    for any `T` takes a `cmath.complex` with parameter `T` and returns a `T`.
  }];

  let arguments = (ins StrAttr:$base_type, Variadic<IRDL_ConstraintType>:$args);
  let results = (outs IRDL_ConstraintType:$output);
  let assemblyFormat = [{ `:` $base_type (`<` $args^ `>`)? attr-dict }];
}

def IRDL_AnyType : IRDL_ConstraintOp<"any_type", [ParentOneOf<["TypeOp", "OperationOp"]>]> {
  let summary = "Constraints to any type";
  let description = [{
    `irdl.any_type` defines a constraint that accepts any type.

    Example:

    ```mlir
    irdl.dialect cmath {
      irdl.type complex_flexible {
        %0 = irdl.any_type
        irdl.parameters(%0)
      }
    }
    ```

    The above program defines a type `complex_flexible` inside the dialect `cmath` that
    has one type parameter that can be any type.
  }];

  let results = (outs IRDL_ConstraintType:$output);
  let assemblyFormat = [{ attr-dict }];
}

def IRDL_AnyOf : IRDL_ConstraintOp<"any_of", [ParentOneOf<["TypeOp", "OperationOp"]>]> {
  let summary = "Constraints to any of the provided constraints";
  let description = [{
    `irdl.any_of` defines a constraint that accepts any type that
    satisfies at least one of its provided type constraints.

    Example:

    ```mlir
    irdl.dialect cmath {
      irdl.type complex {
        %0 = irdl.is_type : i32
        %1 = irdl.is_type : i64
        %2 = irdl.is_type : f32
        %3 = irdl.is_type : f64
        %4 = irdl.any_of(%0, %1, %2, %3)
        irdl.parameters(%4)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath` that
    can has one type parameter that can be either `i32`, `i64`, `f32` or `f32`.
  }];

  let arguments = (ins Variadic<IRDL_ConstraintType>:$args);
  let results = (outs IRDL_ConstraintType:$output);
  let assemblyFormat = [{ `(` $args `)` attr-dict }];
}

def IRDL_And : IRDL_ConstraintOp<"and", [ParentOneOf<["TypeOp", "OperationOp"]>]> {
  let summary = "Constraints to all the provided constraints";
  let description = [{
    `irdl.and` defines a constraint that accepts any type that
    satisfies all of its provided type constraints.

    Example:

    ```mlir
    irdl.dialect cmath {
      irdl.type complex_f32 {
        %0 = irdl.is_type : i32
        %1 = irdl.is_type : f32
        %2 = irdl.any_of(%0, %1) // is 32-bit

        %3 = irdl.is_type : f32
        %4 = irdl.is_type : f64
        %5 = irdl.any_of(%3, %4) // is a float

        %6 = irdl.and(%2, %5) // is a 32-bit float
        irdl.parameters(%6)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath` that
    can has one type parameter that must be 32-bit long and a float (in other
    words, that must be `f32`).
  }];

  let arguments = (ins Variadic<IRDL_ConstraintType>:$args);
  let results = (outs IRDL_ConstraintType:$output);
  let assemblyFormat = [{ `(` $args `)` attr-dict }];
}

#endif // MLIR_DIALECT_IRDL_IR_IRDLOPS
