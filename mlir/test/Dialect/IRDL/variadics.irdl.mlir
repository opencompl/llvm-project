// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: irdl.dialect @testvar {
irdl.dialect @testvar {

  // CHECK-LABEL: irdl.operation @single_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    irdl.operands {
  // CHECK-NEXT:      "single" = %[[v0]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  irdl.operation @single_operand {
    %0 = irdl.is i32
    irdl.operands {
      "single" = single %0
    }
  }

  // CHECK-LABEL: irdl.operation @var_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64
  // CHECK-NEXT:    irdl.operands {
  // CHECK-NEXT:      "single1" = %[[v0]],
  // CHECK-NEXT:      "var" = variadic %[[v1]],
  // CHECK-NEXT:      "single2" = %[[v2]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  irdl.operation @var_operand {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.operands {
      "single1" = %0,
      "var" = variadic %1,
      "single2" = %2
    }
  }

  // CHECK-LABEL: irdl.operation @opt_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64
  // CHECK-NEXT:    irdl.operands {
  // CHECK-NEXT:      "single1" = %[[v0]],
  // CHECK-NEXT:      "optional" = optional %[[v1]],
  // CHECK-NEXT:      "single2" = %[[v2]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  irdl.operation @opt_operand {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.operands {
      "single1" = %0,
      "optional" = optional %1,
      "single2" = %2
    }
  }

  // CHECK-LABEL: irdl.operation @var_and_opt_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64
  // CHECK-NEXT:    irdl.operands {
  // CHECK-NEXT:      "variadic" = variadic %[[v0]],
  // CHECK-NEXT:      "optional" = optional %[[v1]],
  // CHECK-NEXT:      "single" = %[[v2]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  irdl.operation @var_and_opt_operand {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.operands {
      "variadic" = variadic %0,
      "optional" = optional %1,
      "single" = %2
    }
  }

  // CHECK-LABEL: irdl.operation @single_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    irdl.results {
  // CHECK-NEXT:      "single" = %[[v0]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  irdl.operation @single_result {
    %0 = irdl.is i32
    irdl.results {
      "single" = single %0
    }
  }

  // CHECK-LABEL: irdl.operation @var_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64
  // CHECK-NEXT:    irdl.results {
  // CHECK-NEXT:      "single1" = %[[v0]],
  // CHECK-NEXT:      "var" = variadic %[[v1]],
  // CHECK-NEXT:      "single2" = %[[v2]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  irdl.operation @var_result {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.results {
      "single1" = %0,
      "var" = variadic %1,
      "single2" = %2
    }
  }

  // CHECK-LABEL: irdl.operation @opt_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64
  // CHECK-NEXT:    irdl.results {
  // CHECK-NEXT:      "single1" = %[[v0]],
  // CHECK-NEXT:      "optional" = optional %[[v1]],
  // CHECK-NEXT:      "single2" = %[[v2]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  irdl.operation @opt_result {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.results {
      "single1" = %0,
      "optional" = optional %1,
      "single2" = %2
    }
  }

  // CHECK-LABEL: irdl.operation @var_and_opt_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64
  // CHECK-NEXT:    irdl.results {
  // CHECK-NEXT:      "variadic" = variadic %[[v0]],
  // CHECK-NEXT:      "optional" = optional %[[v1]],
  // CHECK-NEXT:      "single" = %[[v2]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  irdl.operation @var_and_opt_result {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.results {
      "variadic" = variadic %0,
      "optional" = optional %1,
      "single" = %2
    }
  }

  // CHECK-LABEL: irdl.operation @var_attr {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64
  // CHECK-NEXT:    irdl.attributes {
  // CHECK-NEXT:      "optional" = optional %[[v0]],
  // CHECK-NEXT:      "single" = %[[v1]],
  // CHECK-NEXT:      "single_no_word" = %[[v2]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  irdl.operation @var_attr {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.attributes {
      "optional" = optional %0,
      "single" = single %1,
      "single_no_word" = %2
    }
  }

  // CHECK-LABEL: irdl.operation @var_region {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.region
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.region
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.region
  // CHECK-NEXT:    irdl.regions {
  // CHECK-NEXT:      "variadic" = variadic %[[v0]],
  // CHECK-NEXT:      "single_no_word" = %[[v1]],
  // CHECK-NEXT:      "single" = %[[v2]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  irdl.operation @var_region {
    %0 = irdl.region
    %1 = irdl.region
    %2 = irdl.region
    irdl.regions {
      "variadic" = variadic %0,
      "single_no_word" = %1,
      "single" = single %2
    }
  }
}
