//===- AttributeWrapper.h - IRDL type wrapper definition --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares wrappers around attribute and type definitions to
// manipulate them in a unified way, using their names and a list of parameters
// encoded as a list of attributes. These wrappers are necessary for IRDL, since
// attributes and types don't have names, nor a way to interact with them in a
// generic way.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_ATTRIBUTEWRAPPER_H_
#define MLIR_DIALECT_IRDL_ATTRIBUTEWRAPPER_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {
namespace irdl {

/// A wrapper around an attribute definition to manipulate its attribute
/// instances in a unified way, using a name and a list of parameters encoded as
/// attributes.
///
/// If the attribute is defined in C++, CppAttributeWrapper should be preferred.
class AttributeWrapper {
public:
  virtual ~AttributeWrapper(){};

  /// Return the unique identifier of the attribute definition.
  virtual TypeID getTypeID() = 0;

  /// Check if the given attribute is an instance of the one wrapped.
  virtual bool isCorrectAttribute(Attribute attr) = 0;

  /// Get the parameters of an attribute. The attribute is expected to be an
  /// instance of the wrapped attribute, which is checked with
  /// `isCorrectAttribute`.
  virtual SmallVector<Attribute> getAttributeParameters(Attribute attr) = 0;

  /// Return the attribute definition name, including the dialect prefix.
  virtual StringRef getName() = 0;

  /// Instantiate the attribute from parameters.
  /// It is expected that the amount of parameters is correct, which is checked
  /// with `getParameterAmount`.
  virtual Attribute
  instantiateAttribute(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<Attribute> parameters) = 0;

  /// Return the amount of parameters the attribute expects.
  virtual size_t getParameterAmount() = 0;
};

/// A wrapper around a type definition to manipulate its type instances
/// in an unified way, using a name and a list of parameters encoded as
/// attributes. The wrappers also acts as an attribute wrapper, and expects
/// the type to be nested in a `TypeAttr`.
///
/// If the type is defined as a C++ class, CppTypeWrapper should be preferred.
class TypeWrapper : public AttributeWrapper {
public:
  virtual ~TypeWrapper(){};

  /// Check if the given type is an instance of the one wrapped.
  virtual bool isCorrectType(Type t) = 0;

  /// Get the parameters of a type. The type is expected to be an instance of
  /// the wrapped type, which is checked with `isCorrectType`.
  virtual SmallVector<Attribute> getTypeParameters(Type t) = 0;

  /// Instantiate the type from parameters.
  /// It is expected that the amount of parameters is correct, which is checked
  /// with `getParameterAmount`.
  virtual Type instantiateType(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<Attribute> parameters) = 0;

  /// Check if the given attribute is a `TypeAttr`, and that it contains an
  /// instance of the type wrapped.
  bool isCorrectAttribute(Attribute attr) override;

  /// Get the parameters of a type. The type is expected to be nested in a
  /// `TypeAttr`, and to be an instance of the wrapped type, which is checked
  /// with `isCorrectType`.
  SmallVector<Attribute> getAttributeParameters(Attribute attr) override;

  /// Instantiate the type from parameters, and wrap it in a `TypeAttr`.
  /// It is expected that the amount of parameters is correct, which is checked
  /// with `getParameterAmount`.
  Attribute instantiateAttribute(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<Attribute> parameters) override;
};

using AttributeWrapperPtr = AttributeWrapper *;
using TypeWrapperPtr = TypeWrapper *;

/// A wrapper around an attribute definition defined with a C++ class to
/// manipulate its attribute instances in a unified way, using a name and a list
/// of parameters encoded as attributes.
template <typename A>
class CppAttributeWrapper : public AttributeWrapper {
public:
  TypeID getTypeID() override { return TypeID::get<A>(); }

  /// Get the parameters of an attribute.
  virtual SmallVector<Attribute> getAttributeParameters(A attr) = 0;

  SmallVector<Attribute> getAttributeParameters(Attribute attr) override {
    return getAttributeParameters(cast<A>(attr));
  };

  bool isCorrectAttribute(Attribute attr) override { return isa<A>(attr); }
};

/// A wrapper around a type definition defined with a C++ class to
/// manipulate its type instances in a unified way, using a name and a list
/// of parameters encoded as attributes.
template <typename T>
class CppTypeWrapper : public TypeWrapper {
public:
  TypeID getTypeID() override { return TypeID::get<T>(); }

  /// Get the parameters of a type.
  virtual SmallVector<Attribute> getTypeParameters(T t) = 0;

  SmallVector<Attribute> getTypeParameters(Type type) override {
    return getTypeParameters(cast<T>(type));
  };

  bool isCorrectType(Type type) override { return isa<T>(type); }
};

} // namespace irdl
} // namespace mlir

#endif // MLIR_DIALECT_IRDL_ATTRIBUTEWRAPPER_H_
