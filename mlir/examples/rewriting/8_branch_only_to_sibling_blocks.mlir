// RUN: mlir-opt %s  -allow-unregistered-dialect -test-pdl-bytecode-pass

module @patterns {
  // Constructing a rewrite that violates block label scoping seems impossible. 
  // PDL has not way to refer to successors, so they can not be matched explicitly.
  // While e.g. an cf.br can be matched, there is no way to nest it into a deeper level to violate scoping or
  // change the successor.
}

module @ir attributes {} {
  func.func @builtin_ops() -> () {
    ^entry():
      cf.br ^bb1
    
    ^bbtest:
      "custom.term"() : () -> ()

    ^bb1():
      %result = "custom.op_with_region"() [^bb2] ({
        ^bb4():
          "custom.term"() : () -> ()
        "custom.term"() : () -> ()
      }) : () -> (i32)
      // cf.br ^bb2

    ^bb3():
      "custom.term"() : () -> ()
      cf.br ^bb2

    ^bb2():
      return
    
  }
}