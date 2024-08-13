// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: irdl.dialect @testd {
irdl.dialect @testd {
  // CHECK: irdl.type @parametric {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   irdl.parameters(%[[v0]])
  // CHECK: }
  irdl.type @parametric {
    %0 = irdl.any
    irdl.parameters(%0)
  }

  // CHECK: irdl.attribute @parametric_attr {
  // CHECK:  %[[v0:[^ ]*]] = irdl.any
  // CHECK:  irdl.parameters(%[[v0]])
  // CHECK: }
  irdl.attribute @parametric_attr {
    %0 = irdl.any
    irdl.parameters(%0)
  }

  // CHECK: irdl.type @attr_in_type_out {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   irdl.parameters(%[[v0]])
  // CHECK: }
  irdl.type @attr_in_type_out {
    %0 = irdl.any
    irdl.parameters(%0)
  }

  // CHECK: irdl.operation @eq {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   irdl.results {
  // CHECK:     "out" = %[[v0]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @eq {
    %0 = irdl.is i32
    irdl.results {
      "out" = %0
    }
  }

  // CHECK: irdl.operation @anyof {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   irdl.results {
  // CHECK:     "out" = %[[v2]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @anyof {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    irdl.results {
      "out" = %2
    }
  }

  // CHECK: irdl.operation @all_of {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   %[[v3:[^ ]*]] = irdl.all_of(%[[v2]], %[[v1]])
  // CHECK:   irdl.results {
  // CHECK:     "out" = %[[v3]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @all_of {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    %3 = irdl.all_of(%2, %1)
    irdl.results {
      "out" = %3
    }
  }

  // CHECK: irdl.operation @any {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   irdl.results {
  // CHECK:     "out" = %[[v0]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @any {
    %0 = irdl.any
    irdl.results {
      "out" = %0
    }
  }

  // CHECK: irdl.operation @dyn_type_base {
  // CHECK:   %[[v1:[^ ]*]] = irdl.base @testd::@parametric
  // CHECK:   irdl.results {
  // CHECK:     "out" = %[[v1]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @dyn_type_base {
    %0 = irdl.base @testd::@parametric
    irdl.results {
      "out" = %0
    }
  }

  // CHECK: irdl.operation @dyn_attr_base {
  // CHECK:   %[[v1:[^ ]*]] = irdl.base @testd::@parametric_attr
  // CHECK:   irdl.attributes {
  // CHECK:     "attr1" = %[[v1]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @dyn_attr_base {
    %0 = irdl.base @testd::@parametric_attr
    irdl.attributes {
      "attr1" = %0
    }
  }

  // CHECK: irdl.operation @named_type_base {
  // CHECK:   %[[v1:[^ ]*]] = irdl.base "!builtin.integer"
  // CHECK:   irdl.results {
  // CHECK:     "out" = %[[v1]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @named_type_base {
    %0 = irdl.base "!builtin.integer"
    irdl.results {
      "out" = %0
    }
  }

  // CHECK: irdl.operation @named_attr_base {
  // CHECK:   %[[v1:[^ ]*]] = irdl.base "#builtin.integer"
  // CHECK:   irdl.attributes {
  // CHECK:     "attr1" = %[[v1]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @named_attr_base {
    %0 = irdl.base "#builtin.integer"
    irdl.attributes {
      "attr1" = %0
    }
  }

  // CHECK: irdl.operation @dynparams {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   %[[v3:[^ ]*]] = irdl.parametric @testd::@parametric<%[[v2]]>
  // CHECK:   irdl.results {
  // CHECK:     "out" = %[[v3]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @dynparams {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    %3 = irdl.parametric @testd::@parametric<%2>
    irdl.results {
      "out" = %3
    }
  }

  // CHECK: irdl.operation @constraint_vars {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   irdl.results {
  // CHECK:     "out1" = %[[v2]],
  // CHECK:     "out2" = %[[v2]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @constraint_vars {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    irdl.results {
      "out1" = %2,
      "out2" = %2
    }
  }

  // CHECK: irdl.operation @attrs {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   irdl.attributes {
  // CHECK:     "attr1" = %[[v0]],
  // CHECK:     "attr2" = %[[v1]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @attrs {
    %0 = irdl.is i32
    %1 = irdl.is i64

    irdl.attributes {
      "attr1" = %0,
      "attr2" = %1
    }
  }
  // CHECK: irdl.operation @regions {
  // CHECK:   %[[r0:[^ ]*]] = irdl.region
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[r1:[^ ]*]] = irdl.region(%[[v0]], %[[v1]])
  // CHECK:   %[[r2:[^ ]*]] = irdl.region with size 3
  // CHECK:   %[[r3:[^ ]*]] = irdl.region()
  // CHECK:   irdl.regions {
  // CHECK:     "any" = %[[r0]],
  // CHECK:     "two_args" = %[[r1]],
  // CHECK:     "three_blocks" = %[[r2]],
  // CHECK:     "no_args" = %[[r3]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @regions {
    %r0 = irdl.region
    %v0 = irdl.is i32
    %v1 = irdl.is i64
    %r1 = irdl.region(%v0, %v1)
    %r2 = irdl.region with size 3
    %r3 = irdl.region()

    irdl.regions {
      "any" = %r0,
      "two_args" = %r1,
      "three_blocks" = %r2,
      "no_args" = %r3
    }
  }

  // CHECK: irdl.operation @region_and_operand {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   %[[r0:[^ ]*]] = irdl.region(%[[v0]])
  // CHECK:   irdl.operands {
  // CHECK:     "in" = %[[v0]]
  // CHECK:   }
  // CHECK:   irdl.regions {
  // CHECK:     "region" = %[[r0]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @region_and_operand {
    %v0 = irdl.any
    %r0 = irdl.region(%v0)

    irdl.operands {
      "in" = %v0
    }
    irdl.regions {
      "region" = %r0
    }
  }

  // CHECK: irdl.operation @element_type {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.base "!builtin.vector"
  // CHECK:   %[[v2:[^ ]*]] = irdl.has_element_type %[[v0]]
  // CHECK:   %[[v3:[^ ]*]] = irdl.all_of(%[[v1]], %[[v2]])
  // CHECK:   irdl.operands {
  // CHECK:     "in" = %[[v3]]
  // CHECK:   }
  // CHECK:   irdl.results {
  // CHECK:     "out" = %[[v0]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @element_type {
    %0 = irdl.is i32
    %1 = irdl.base "!builtin.vector"
    %2 = irdl.has_element_type %0
    %3 = irdl.all_of(%1, %2)
    irdl.operands {
      "in" = %3
    }
    irdl.results {
      "out" = %0
    }
  }

  // CHECK: irdl.operation @rank {
  // CHECK:   %[[v0:[^ ]*]] = irdl.base "!builtin.tensor"
  // CHECK:   %[[v1:[^ ]*]] = irdl.has_rank 2
  // CHECK:   %[[v2:[^ ]*]] = irdl.all_of(%[[v0]], %[[v1]])
  // CHECK:   irdl.operands {
  // CHECK:     "in" = %[[v2]]
  // CHECK:   }
  // CHECK:   irdl.results {
  // CHECK:     "out" = %[[v2]]
  // CHECK:   }
  // CHECK: }
  irdl.operation @rank {
    %0 = irdl.base "!builtin.tensor"
    %1 = irdl.has_rank 2
    %2 = irdl.all_of(%0, %1)
    irdl.operands {
      "in" = %2
    }
    irdl.results {
      "out" = %2
    }
  }
}
