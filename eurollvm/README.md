To run the simplify pass:

`mlir-opt --allow-unregistered-dialect <file-name> --affine-simplify-if`

The main implementation file for this is in ../mlir/lib/Dialect/Affine/Transforms/SimplifyAffineIf.cpp

To run the coalesce pass:

`mlir-opt --allow-unregistered-dialect <file-name> --affine-coalesce-memrefs`

The main implementation file for this is in ../mlir/lib/Dialect/Affine/Transforms/CoalesceMemRefs.cpp
