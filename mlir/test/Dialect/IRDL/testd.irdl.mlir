// RUN: mlir-irdl-opt %s | mlir-irdl-opt | FileCheck %s

// CHECK: irdl.dialect @testd {
irdl.dialect @testd {

  // CHECK: irdl.type @parametric {
  // CHECK:   irdl.parameters(param: Any)
  // CHECK: }
  irdl.type @parametric {
    irdl.parameters(param: Any)
  }

  // CHECK: irdl.operation eq {
  // CHECK:   irdl.results(res: i32)
  // CHECK: }  
  irdl.operation eq {
    irdl.results(res: i32)
  }

  // CHECK: irdl.operation anyof {
  // CHECK:   irdl.results(res: AnyOf<i32, i64>)
  // CHECK: }
  irdl.operation anyof {
    irdl.results(res: AnyOf<i32, i64>)
  }

  // CHECK: irdl.operation and {
  // CHECK:   irdl.results(res: And<AnyOf<i32, i64>, i64>)
  // CHECK: }
  irdl.operation and {
    irdl.results(res: And<AnyOf<i32, i64>, i64>)
  }

  // CHECK: irdl.operation any {
  // CHECK:   irdl.results(res: Any)
  // CHECK: }
  irdl.operation any {
    irdl.results(res: Any)
  }

  // CHECK: irdl.operation dynbase {
  // CHECK:   irdl.results(res: testd.parametric)
  // CHECK: }
  irdl.operation dynbase {
    irdl.results(res: testd.parametric)
  }

  // CHECK: irdl.operation base {
  // CHECK:   irdl.results(res: builtin.complex)
  // CHECK: }
  irdl.operation base {
    irdl.results(res: builtin.complex)
  }

  // CHECK: irdl.operation dynparams {
  // CHECK:   irdl.results(res: testd.parametric<AnyOf<i32, i64>>)
  // CHECK: }
  irdl.operation dynparams {
    irdl.results(res: testd.parametric<AnyOf<i32, i64>>)
  }

  // CHECK: irdl.operation params {
  // CHECK:   irdl.results(res: builtin.complex<AnyOf<i32, i64>>)
  // CHECK: }
  irdl.operation params {
    irdl.results(res: builtin.complex<AnyOf<i32, i64>>)
  }

  // CHECK: irdl.operation constraint_vars {
  // CHECK:   irdl.constraint_vars(T: AnyOf<i32, i64>)
  // CHECK:   irdl.results(res1: ?T, res2: ?T)
  // CHECK: }
  irdl.operation constraint_vars {
    irdl.constraint_vars(T: AnyOf<i32, i64>)
    irdl.results(res1: ?T, res2: ?T)
  }
}
