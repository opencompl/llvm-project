Simplify pass:
  
mlir-opt --allow-unregistered-dialect <file-name> --affine-simplify-if

Coalesce pass:

mlir-opt --allow-unregistered-dialect <file-name> --affine-coalesce-memrefs
