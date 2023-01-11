// RUN: mlir-irdl-opt %s | mlir-irdl-opt | FileCheck %s

irdl.dialect testd {
  // CHECK: irdl.type parametric {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any_type
  // CHECK:   irdl.parameters(%[[v0]])
  // CHECK: }
  irdl.type parametric {
    %0 = irdl.any_type
    irdl.parameters(%0)
  }

  // CHECK: irdl.operation eq {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is_type : i32
  // CHECK:   irdl.results(%[[v0]])
  // CHECK: }
  irdl.operation eq {
    %0 = irdl.is_type : i32
    irdl.results(%0)
  }

  // CHECK: irdl.operation eq_param {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is_type : "testd.parametric"<i32>
  // CHECK:   irdl.results(%[[v0]])
  // CHECK: }
  irdl.operation eq_param {
    %0 = irdl.is_type : "testd.parametric"<i32>
    irdl.results(%0)
  }

  // CHECK: irdl.operation anyof {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is_type : i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is_type : i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   irdl.results(%[[v2]])
  // CHECK: }
  irdl.operation anyof {
    %0 = irdl.is_type : i32
    %1 = irdl.is_type : i64
    %2 = irdl.any_of(%0, %1)
    irdl.results(%2)
  }

  // CHECK: irdl.operation all_of {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is_type : i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is_type : i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   %[[v3:[^ ]*]] = irdl.all_of(%[[v2]], %[[v1]])
  // CHECK:   irdl.results(%[[v3]])
  // CHECK: }
  irdl.operation all_of {
    %0 = irdl.is_type : i32
    %1 = irdl.is_type : i64
    %2 = irdl.any_of(%0, %1)
    %3 = irdl.all_of(%2, %1)
    irdl.results(%3)
  }

  // CHECK: irdl.operation any {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any_type
  // CHECK:   irdl.results(%[[v0]])
  // CHECK: }
  irdl.operation any {
    %0 = irdl.any_type
    irdl.results(%0)
  }

  // CHECK: irdl.operation dynbase {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any_type
  // CHECK:   %[[v1:[^ ]*]] = irdl.parametric_type : "testd.parametric"<%[[v0]]>
  // CHECK:   irdl.results(%[[v1]])
  // CHECK: }
  irdl.operation dynbase {
    %0 = irdl.any_type
    %1 = irdl.parametric_type : "testd.parametric"<%0>
    irdl.results(%1)
  }

  // CHECK: irdl.operation dynparams {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is_type : i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is_type : i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   %[[v3:[^ ]*]] = irdl.parametric_type : "testd.parametric"<%[[v2]]>
  // CHECK:   irdl.results(%[[v3]])
  // CHECK: }
  irdl.operation dynparams {
    %0 = irdl.is_type : i32
    %1 = irdl.is_type : i64
    %2 = irdl.any_of(%0, %1)
    %3 = irdl.parametric_type : "testd.parametric"<%2>
    irdl.results(%3)
  }

  // CHECK: irdl.operation params {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is_type : i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is_type : i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   %[[v3:[^ ]*]] = irdl.parametric_type : "builtin.complex"<%[[v2]]>
  // CHECK:   irdl.results(%[[v3]])
  // CHECK: }
  irdl.operation params {
    %0 = irdl.is_type : i32
    %1 = irdl.is_type : i64
    %2 = irdl.any_of(%0, %1)
    %3 = irdl.parametric_type : "builtin.complex"<%2>
    irdl.results(%3)
  }

  // CHECK: irdl.operation constraint_vars {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is_type : i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is_type : i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   irdl.results(%[[v2]], %[[v2]])
  // CHECK: }
  irdl.operation constraint_vars {
    %0 = irdl.is_type : i32
    %1 = irdl.is_type : i64
    %2 = irdl.any_of(%0, %1)
    irdl.results(%2, %2)
  }
}
