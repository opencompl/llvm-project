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
    IRDL is an based backend for IRDL. It allows encoding IRDL
    dialect constraints in form, allowing easier analysis and
    optimization.

    IRDL code is meant to be generated and not written by hand.
    As such, the design focuses on ease of generation/analysis instead
    of ease of writing/reading.

    Operation `irdl.dialect` defines a dialect, `irdl.type` defines
    a type and `irdl.operation` defines an operation. Constraints on
    type variables are also expressed using operations.

    A dialect example is shown below:

    ```mlir
    irdl.dialect cmath {
      irdl.type complex {
        %0 = irdl.is_type : f32
        %1 = irdl.is_type : f64
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(%2)
      }

      irdl.operation mul {
        %0 = irdl.is_type : f32
        %1 = irdl.is_type : f64
        %2 = irdl.any_of(%0, %1)
        %3 = irdl.parametric_type : "cmath.complex"<%2>
        irdl.operands(%3, %3)
        irdl.results(%3)
      }
    }
    ```

    This program defines a `cmath` dialect that defines a `complex` type, and
    a `mul` operation. Both express constraints over their parameters using
    SSA constraint operations. Informally, one can see those SSA values as
    constraint variables that evaluate to a single type at constraint
    evaluation. For example, the result of the `irdl.any_of` stored in `%2`
    in the `mul` operation will collapse into either `f32` or `f64` for the
    entirety of this instance of `mul` constraint evaluation. As such,
    both operands and the result of `mul` must be of equal type (and not just
    satisfy the same constraint).
  }];

  let useDefaultTypePrinterParser = 1;
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
    ::mlir::irdl::TypeWrapper* getTypeWrapper(StringRef typeName);

    /// IRDL context.
    ::mlir::irdl::IRDLContext irdlContext;
  private:
    /// Add a type wrapper to IRDL.
    void addTypeWrapper(std::unique_ptr<::mlir::irdl::TypeWrapper> wrapper);
  }];

  let name = "irdl";
  let cppNamespace = "::mlir::irdl";
}

#endif // MLIR_DIALECT_IRDL_IR_IRDL
