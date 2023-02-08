// RUN: mlir-opt %s  -allow-unregistered-dialect -test-pdl-bytecode-pass -split-input-file

module @patterns {
  // This rewrite removes the operation that uses the op custom.const and the const itself
  pdl.pattern : benefit(3) {
    %const_type = pdl.type
    %attr0 = pdl.attribute
    %cst1 = pdl.operation "custom.const" {"value" = %attr0} -> (%const_type : !pdl.type)
    %val0 = pdl.result 0 of %cst1

    %cst_user = pdl.operation (%val0 : !pdl.value)

    pdl.rewrite %cst_user {
      pdl.operation "custom.tmp" // placeholder because this region must not be empty. comment out to try
      // pdl.erase %cst_user // comment in to see the assertion
    }
  }
}

module @ir attributes {} {
  // Both scf.yield are removed using two applications of this rewrite, leaving the IR invalid at two places. This is only catched on verification.
  func.func @builtin_ops() -> (f32) {
    %cond = "custom.const"() {value = true} : () -> i1
    %x = scf.if %cond -> (f32) {
      %x_true = "custom.const"() {value = 9.0 : f32} : () -> f32
      scf.yield %x_true : f32
    } else {
      %x_false = "custom.const"() {value = 9.0 : f32} : () -> f32
      scf.yield %x_false : f32
    }

    return %x : f32
  }
}

// -----

module @patterns {
  // Same rewrite like above: This rewrite removes the operation that uses the op custom.const and the const itself
  pdl.pattern : benefit(3) {
    %const_type = pdl.type
    %attr0 = pdl.attribute
    %cst1 = pdl.operation "custom.const" {"value" = %attr0} -> (%const_type : !pdl.type)
    %val0 = pdl.result 0 of %cst1

    %cst_user = pdl.operation (%val0 : !pdl.value)

    pdl.rewrite %cst_user {
      pdl.operation "custom.tmp" // placeholder because this region must not be empty. comment out to try
      // pdl.erase %cst_user // comment in to see the assertion
    }
  }
}

module @ir attributes {} {
  // As these are unregistered ops, the system has no notion of terminators anymore. If the erasure of custom.const is commented in though,
  // the rewrite happily yields an empty block, which is always invalid. 
  func.func @unregistered_ops() -> (i32) {
    %result = "custom.op_with_region"() ({
        %nested_cst = "custom.const"() {value = 9 : i32} : () -> i32
        "custom.return"(%nested_cst) : (i32) -> ()
    }) : () -> (i32)
    return %result : i32
  }
}