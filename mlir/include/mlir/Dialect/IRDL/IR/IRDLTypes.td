//===- IRDLTypes.td - IRDL Types ---------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types IRDL uses.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDLTYPES
#define MLIR_DIALECT_IRDL_IR_IRDLTYPES

include "mlir/IR/AttrTypeBase.td"
include "IRDL.td"

class IRDL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<IRDL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IRDL_ConstraintType : IRDL_Type<"Constraint", "constr"> {
  let summary = "Type representing a type constraint";
  let description = [{
    The Constraint type is the type of type constraints when
    used in SSA form.

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

    Here, `%0`, `%1` and `%2` are all of type `Constraint`.
  }];
}

#endif // MLIR_DIALECT_IRDL_IR_IRDLTYPES
