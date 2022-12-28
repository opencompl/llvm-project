// RUN: mlir-irdl-opt %s | mlir-irdl-opt | FileCheck %s

module {
  // CHECK-LABEL: irdl.dialect @testd {
  irdl.dialect @testd {
    // CHECK: irdl.type singleton
    irdl.type singleton

    // CHECK: irdl.type parametrized {
    // CHECK:   irdl.parameters(arg1: Any, arg2: AnyOf<i32, i64>)
    // CHECK: }
    irdl.type parametrized {
      irdl.parameters(arg1: Any, arg2: AnyOf<i32, i64>)
    }

    // CHECK: irdl.operation any {
    // CHECK:   irdl.results(res: Any)
    // CHECK: }
    irdl.operation any {
      irdl.results(res: Any)
    }
  }
}
