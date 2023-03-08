//===- IRDLAttributes.td - IR Definition Language Dialect --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes used in IRDL.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDLATTRIBUTES
#define MLIR_DIALECT_IRDL_IR_IRDLATTRIBUTES

include "IRDL.td"
include "mlir/IR/AttrTypeBase.td"

class IRDL_AttrDef<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<IRDL_Dialect, name, traits, "::mlir::Attribute"> {
  let mnemonic = attrMnemonic;
}

def TypeOrAttrDefRefAttr : IRDL_AttrDef<"TypeOrAttrDefRef",
                                        "type_or_attr_def_ref"> {
  let summary = "reference to a type or attribute definition";
  let description = [{
    An `irdl.type_or_attr_def_ref` references a type or attribute definition.
    A type or attribute definition can either be a `TypeWrapper *` or
    an `AttrWrapper *`, which refers to a C++-defined type or attribute,
    or a `SymbolRefAttr` which refers to a type or attribute defined with IRDL.
  }];
  let parameters =
    (ins OptionalParameter<"mlir::irdl::TypeWrapperPtr">:$typeWrapper,
         OptionalParameter<"mlir::irdl::AttributeWrapperPtr">:$attrWrapper,
         OptionalParameter<"mlir::SymbolRefAttr">:$symRef);

  let builders = [
    AttrBuilder<(ins "mlir::irdl::TypeWrapper*":$typeWrapper), [{
      return $_get($_ctxt, typeWrapper, nullptr, mlir::SymbolRefAttr());
    }]>,
    AttrBuilder<(ins "mlir::irdl::AttributeWrapper*":$attrWrapper), [{
      return $_get($_ctxt, nullptr, attrWrapper, mlir::SymbolRefAttr());
    }]>,
    AttrBuilderWithInferredContext<(ins "mlir::SymbolRefAttr":$symRef), [{
      return $_get(symRef.getContext(), nullptr, nullptr, symRef);
    }]>
  ];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif // MLIR_DIALECT_IRDL_IR_IRDLATTRIBUTES
