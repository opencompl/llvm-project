# Extensible dialects

This file documents the design and API of the extensible dialects. Extensible
dialects are dialects that can be extended with new operations and types defined
at runtime. This let users define dialects with metaprogramming, or from another
language, without having to recompile C++ code.

[TOC]

## Usage

### Making a dialect extensible at runtime

Dialects defined in C++ can be extended with new operations and types at runtime
by making them inherit from `mlir::ExtensibleDialect` instead of `mlir::Dialect`
(note that `ExtensibleDialect` inherits from `Dialect`). The `ExtensibleDialect`
class contains the necessary fields and methods to extend the dialect at
runtime with new types and operations.

```c++
class MyDialect : public mlir::ExtensibleDialect {
    ...
}
```

For dialects defined in TableGen, this is done by setting the `isExtensible`
flag to `1`.

```tablegen
def Test_Dialect : Dialect {
  let isExtensible = 1;
  ...
}
```

Extensible `Dialect` can be casted back to `ExtensibleDialect` using
`llvm::dyn_cast`, or `llvm::cast`:

```c++
if (auto extensibleDialect = llvm::dyn_cast<ExtensibleDialect>(dialect)) {
    ...
}
```

### Defining an operation at runtime

The `DynamicOpDefinition` class represents the definition of an operation
defined at runtime. It is created using the `DynamicOpDefinition::get`
functions. An operation defined at runtime needs to define a name, a dialect in
which the operation will be registered in, an operation verifier. It can also
define optionally a custom parser and a printer, an operation fold hook, and a
function that returns the canonicalization patterns.

```c++
// The operation name, without the dialect name prefix.
StringRef name = "my_operation_name";

// The dialect defining the operation.
Dialect* dialect = ctx->getOrLoadDialect<MyDialect>();

// Operation verifier definition.
AbstractOperation::VerifyInvariantsFn verifyFn = [](Operation* op) {
    // Logic for the operation verification.
    ...
}

// Parser function definition.
AbstractOperation::ParseAssemblyFn parseFn =
    [](OpAsmParser &parser, OperationState &state) {
        // Parse the operation, given that the name is already parsed.
        ...    
};

// Printer function
auto printFn = [](Operation *op, OpAsmPrinter &printer) {
        printer << op->getName();
        // Print the operation, given that the name is already printed.
        ...
};

// General folder implementation, see AbstractOperation::foldHook for more
// information.
auto foldHookFn = [](Operation * op, ArrayRef<Attribute> operands, 
                                   SmallVectorImpl<OpFoldResult> &result) {
    ...
};

// Returns any canonicalization pattern rewrites that the operation
// supports, for use by the canonicalization pass.
auto getCanonicalizationPatterns = 
        [](RewritePatternSet &results, MLIRContext *context) {
    ...
}

// Definition of the operation.
std::unique_ptr<DynamicOpDefinition> opDef =
    DynamicOpDefinition::get(name, dialect, std::move(verifyFn),
        std::move(parseFn), std::move(printFn), std::move(foldHookFn),
        std::move(getCanonicalizationPatterns));
```

Once the operation is defined, it can be registered by an `ExtensibleDialect`:

```c++
extensibleDialect->addDynamicOperation(std::move(opDef));
```

Note that the `Dialect` given to the operation should be the one registering
the operation.

### Using an operation defined at runtime

It is possible to match on an operation defined at runtime using their names:

```c++
if (op->getName().getStringRef() == "my_dialect.my_dynamic_op") {
    ...
}
```

An operation defined at runtime can be created by creating an `OperationState`
with its name, and passing it to a rewriter such as  a `PatternRewriter`.

```c++
OperationState state(location, "my_dialect.my_dynamic_op",
                     operands, resultTypes, attributes);

rewriter.createOperation(state);
```


### Defining a type at runtime

Contrary to types defined in C++ or in TableGen, types defined at runtime can
only have as argument a list of `Attribute`.

Similarily to operations, a type is defined at runtime using the class
`DynamicTypeDefinition`, which is created using the `DynamicTypeDefinition::get`
functions. A type definition requires a name, the dialect that will register the
type, and a parameter verifier. It can also define optionally a custom parser
and printer for the arguments (the type name is assumed to be already
parsed/printed).

```c++
// The type name, without the dialect name prefix.
StringRef name = "my_type_name";

// The dialect defining the operation.
Dialect* dialect = ctx->getOrLoadDialect<MyDialect>();

// The type verifier. A type defined at runtime has a list of attributes as parameters.
auto verifier = [](function_ref<InFlightDiagnostic()> emitError,
         ArrayRef<Attribute> args) {
    ...
};

// The type parameters parser.
auto parser = [](DialectAsmParser &parser,
        llvm::SmallVectorImpl<Attribute> &parsedParams) {
    ...
};

// The type parameters printer.
auto printer =[](DialectAsmPrinter &printer, ArrayRef<Attribute> params) {
    ...
};

std::unique_ptr<DynamicTypeDefinition> typeDef =
    DynamicTypeDefinition::get(std::move(name), std::move(dialect),
            std::move(verifier), std::move(printer), std::move(parser)); 
```

If the printer and the parser are ommited, a default parser and printer is
generated with the format `!dialect.typename<arg1, arg2, ..., argN>`.

The type can then be registered by the `ExtensibleDialect`:

```c++
dialect->addDynamicType(std::move(typeDef));
```

### Parsing types defined at runtime in an extensible dialect

In order to parse types defined at runtime, it is necessary to add in the
`MyDialect::parseType` method the necessary support.

```c++
Type MyDialect::parseType(DialectAsmParser &parser) const {
    ...
    // The type name.
    StringRef typeTag;
    if (failed(parser.parseKeyword(&typeTag)))
        return Type();

    // Try to parse a dynamic type with 'typeTag' name.
    Type dynType;
    auto parseResult = parseOptionalDynamicType(typeTag, parser, dynType);
    if (parseResult.hasValue()) {
        if (succeeded(parseResult.getValue()))
            return dynType;
         return Type();
    }
```

### Using a type defined at runtime

Dynamic types are instances of `DynamicType`. It is possible to get a dynamic
type with `DynamicType::get` and `ExtensibleDialect::lookupTypeDefinition`.

```c++
auto typeDef = extensibleDialect->lookupTypeDefinition("my_dynamic_type");
ArrayRef<Attribute> params = ...;
auto type = DynamicType::get(typeDef, params);
```

It is also possible to cast a `Type` known to be defined at runtime to a
`DynamicType`.

```c++
auto dynType = type.cast<DynamicType>();
auto typeDef = dynType.getTypeDef();
auto args = dynType.getParams();
```

## Implementation details

### Extensible dialect

The role of extensible dialects is to own the necessary data for defined
operations and types. They also contain the necessary accessors to easily
access them.

In order to cast a `Dialect` back to an `ExtensibleDialect`, we implement the
`IsExtensibleDialect` interface to all `ExtensibleDialect`. The casting is done
by checking if the `Dialect` implements `IsExtensibleDialect` or not.

### Operation representation and registration

Operations are represented in mlir using the `AbstractOperation` class. They are
registered in dialects the same way operations defined in C++ are registered,
which is by calling `AbstractOperation::insert`.

The only difference is that a new `TypeID` needs to be created for each
operation, since operations are not represented by a C++ class. This is done
using a `TypeIDAllocator`, which can allocate a new unique `TypeID` at runtime.

### Type representation and registration

Unlike operations, types need to define a C++ storage class that takes care of
type parameters. They also need to define another C++ class to access that
storage. `DynamicTypeStorage` defines the storage of types defined at runtime,
and `DynamicType` gives access to the storage, as well as defining useful
functions. A `DynamicTypeStorage` contains a list of `Attribute` type
parameters, as well as a pointer to the type definition.

Types are registered using the `Dialect::addType` method, which expect a
`TypeID` that is generated using a `TypeIDAllocator`. The type uniquer also
register the type with the given `TypeID`. This mean that we can reuse our
single `DynamicType` with different `TypeID` to represent the different types
defined at runtime.

Since the different types defined at runtime have different `TypeID`, it is not
possible to use `TypeID` to cast a `Type` into a `DynamicType`. Thus, similar to
`Dialect`, all `DynamicType` define a `IsDynamicTypeTrait`, so casting a `Type`
to a `DynamicType` boils down to querying the `IsDynamicTypeTrait` trait.
