// RUN: mlir-opt %s  -allow-unregistered-dialect -test-pdl-bytecode-pass -split-input-file


/////////////////////////////// removing an op that is used somewhere else //////////////////////////////////
module @patterns {
  // This rewrite removes add and cst1, which still has another use! This is caught with an assertion at application time
  pdl.pattern : benefit(1) {
    %const_type = pdl.type
    %attr0 = pdl.attribute
    %cst1 = pdl.operation "custom.const" {"value" = %attr0} -> (%const_type : !pdl.type)
    %val0 = pdl.result 0 of %cst1
    %attr1 = pdl.attribute
    %cst2 = pdl.operation "custom.const" {"value" = %attr1} -> (%const_type : !pdl.type)
    %val1 = pdl.result 0 of %cst2
    %add = pdl.operation "custom.add" (%val0, %val1 : !pdl.value, !pdl.value) -> (%const_type : !pdl.type)

    pdl.rewrite %add {
      pdl.erase %add
      // pdl.erase %cst1 // comment in to see the assertion
    }
  }
}

module @ir attributes {} {
  func.func @inter_block_ssa_dom() -> () {
    %0 = "custom.const"() {value = 1 : i32} : () -> i32
    %1 = "custom.const"() {value = 2 : i32} : () -> i32
    %2 = "custom.add"(%0, %1) : (i32, i32) -> i32
    "custom_user"(%0, %1) : (i32, i32) -> () 
  }
}

// -----

/////////////////////////////// removing an op that is used by what we matched apart //////////////////////////////////
module @patterns {
  // This rewrite removes cst1, which is clearly in use by the add we also matched! This is caught with an assertion at application time
  pdl.pattern : benefit(1) {
    %const_type = pdl.type
    %attr0 = pdl.attribute
    %cst1 = pdl.operation "custom.const" {"value" = %attr0} -> (%const_type : !pdl.type)
    %val0 = pdl.result 0 of %cst1
    %attr1 = pdl.attribute
    %cst2 = pdl.operation "custom.const" {"value" = %attr1} -> (%const_type : !pdl.type)
    %val1 = pdl.result 0 of %cst2
    %add = pdl.operation "custom.add" (%val0, %val1 : !pdl.value, !pdl.value) -> (%const_type : !pdl.type)

    pdl.rewrite %add {
      pdl.erase %add//%cst1 // comment in %cst1 to see the assertion
    }
  }
}

module @ir attributes {} {
  func.func @inter_block_ssa_dom() -> () {
    %0 = "custom.const"() {value = 1 : i32} : () -> i32
    %1 = "custom.const"() {value = 2 : i32} : () -> i32
    %2 = "custom.add"(%0, %1) : (i32, i32) -> i32
  }
}

// -----

/////////////////////////////// adding ops that use the root op, without replacing the root op //////////////////////////////////
module @patterns {

  pdl.pattern : benefit(5) {
    // This rewrite matches the custom.add.
    // The rewrite just adds a new operation before the root op. As new operations are always inserted before the root op, 
    // this use will not be dominated
    
    %const_type = pdl.type
    %val0 = pdl.operand
    %val1 = pdl.operand
    %add = pdl.operation "custom.add" (%val0, %val1 : !pdl.value, !pdl.value) -> (%const_type : !pdl.type)
    %added_val = pdl.result 0 of %add

    pdl.rewrite %add {
      %new_cst = pdl.operation "custom.op"(%added_val : !pdl.value) -> (%const_type : !pdl.type)
      // %new_add = pdl.operation "custom.new_add" (%val0, %val1 : !pdl.value, !pdl.value) -> (%const_type : !pdl.type)
      // %new_val = pdl.result 0 of %new_add
      // pdl.replace %add with (%new_val : !pdl.value)
      // pdl.erase %new_cst // with this commented in the rewrite just does custom.add -> custom.new_add. Comment it out to see the assertion
    }
  }
  // This rewrite just illustrates that this is not caught at verification after the pass but directly after the rewrite above. i.e. we 
  // can not apply subsequent rewrites. So this kind of error seems to be checked differently than the other errors above
  pdl.pattern : benefit(4) {
    %const_type = pdl.type
    %attr0 = pdl.attribute
    %cst1 = pdl.operation "custom.const" {"value" = %attr0} -> (%const_type : !pdl.type)
    %val0 = pdl.result 0 of %cst1
    %attr1 = pdl.attribute
    %cst2 = pdl.operation "custom.const" {"value" = %attr1} -> (%const_type : !pdl.type)
    %val1 = pdl.result 0 of %cst2
    %add = pdl.operation "custom.new_add" (%val0, %val1 : !pdl.value, !pdl.value) -> (%const_type : !pdl.type)

    pdl.rewrite %add {
      %commuted = pdl.operation "custom.commuted_add" (%val1, %val0: !pdl.value, !pdl.value) -> (%const_type : !pdl.type)
      %new_val = pdl.result 0 of %commuted
      pdl.replace %add with (%new_val : !pdl.value)

    }
  }
}

module @ir attributes {} {
  func.func @inter_block_ssa_dom() -> (i32) {
    %0 = "custom.const"() {value = 1 : i32} : () -> i32
    %1 = "custom.const"() {value = 2 : i32} : () -> i32
    %2 = "custom.add"(%0, %1) : (i32, i32) -> i32
    return %2 : i32
  }
}