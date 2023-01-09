//===- IRDL.td - IR Definition Language Dialect ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRDL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDL
#define MLIR_DIALECT_IRDL_IR_IRDL

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IRDL Dialect
//===----------------------------------------------------------------------===//

def IRDL_Dialect : Dialect {
  let summary = "IR Definition Language Dialect";
  let description = [{
    IR Definition Language (IRDL) allows defining dialects in a declarative
    format. Those dialects can be loaded at runtime into MLIR.

    TODO: Motivation

    Dialects expressed in IRDL can register types, attributes and operations.

    ```mlir
    // Defines the `cmath` dialect, representing complex numbers
    // and operations over them.
    irdl.dialect @cmath {
      // Defines a singleton `cmath.complex` type,
      // representing a complex number.
      irdl.type @complex {}

      // Defines a `cmath.norm` operation, representing
      // the norm operator on complex numbers. This operation
      // takes a complex number, and returns an f32.
      irdl.operation @norm {
        irdl.operands(x: cmath.complex)
        irdl.result(res: f32)
      }
    }
    ```

    Types and attributes can be parametrized using other types and attributes.
    In order to limit the parameters a type or attribute can receive, IRDL
    provides a logical constraint system to only accept types matching a
    certain structure.

    ```mlir
    // We now want the `cmath` dialect to represent complex
    // numbers with a parametrized scalar type (either f32
    // or f64).
    irdl.dialect @cmath {
      irdl.type @complex {
        // `cmath.complex` can now be parametrized, with either
        // f32 or f64, yielding the two possible variants
        // `cmath.complex<f32>` and `cmath.complex<f64>`.
        irdl.parameters(scalarType: AnyOf<f32, f64>)
      }
    }
    ```

    Operations can use the same constraint system to generically
    accept different types as operands and results. It is often
    desirable to ensure types and attributes are identical accross
    operands in a specific instanciation of the operation. However,
    while operands can be generic using only structural constraints,
    there is no way to add equality constraints across operands.
    In order to allow this, IRDL also provides constraint variables.
    Constraint variables can be used within constraints and represent
    one specific type or attribute instance. When checking that a type
    or attribute matches a constraint, the IRDL constraint checker
    will try to identify what value for the type variable satisfies
    the constraint, failing if one does not exist.

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex {
        irdl.parameters(scalarType: AnyOf<f32, f64>)
      }

      irdl.operation @norm {
        // We want to make sure the return value of
        // the norm operation is the same as the
        // scalar type parametrizing the instance
        // of `cmath.complex`.

        // To do that, we need to define the constraint
        // variable `T`, along with the types it is
        // allowed to match. It will represent the
        // scalar type used.
        irdl.constraint_vars(T: AnyOf<f32, f64>)

        // Then, we specify where the value of `T`
        // will appear.
        irdl.operands(c: cmath.complex<?T>)
        irdl.results(res: ?T)

        // This will match instances of this operation
        // such as `cmath.norm(f32) -> f32` or
        // `cmath.norm(f64) -> f64` but will refuse
        // `cmath.norm(f32) -> f64` for example.
      }
    }
    ```

    TODO: Show an example of attribute definition.    
  }];

  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
  public:
    /// Register new attributes that can be used by IRDL.
    template <typename Attr>
    void addNewAttribute() {
      addAttributes<Attr>();
    }

    /// Add a concrete type wrapper in IRDL.
    template <typename T>
    void addTypeWrapper() {
         addTypeWrapper(std::make_unique<T>());
    }

    /// Get a type wrapper with the given name.
    /// Returns nullptr if no type wrapper with that name is found.
    TypeWrapper* getTypeWrapper(StringRef typeName);

    /// Add a type wrapper to IRDL.
    void addTypeWrapper(std::unique_ptr<mlir::irdl::TypeWrapper> wrapper);

    /// Context object for this IRDL instance (type definitions).
    mlir::irdl::IRDLContext irdlContext;

  private:  
    /// Register the attributes used by default by IRDL.
    void registerAttributes();
    /// Register the attributes used by standard interfaces wrappers in IRDL.
    void registerStandardInterfaceAttributes();
  }];

  let name = "irdl";
  let cppNamespace = "::mlir::irdl";
}

#endif // MLIR_DIALECT_IRDL_IR_IRDL
