// RUN: mlir-irdl-opt %s | mlir-irdl-opt | FileCheck %s

module {
  // CHECK-LABEL: irdl.dialect @cmath {
  irdl.dialect @cmath {

    // CHECK: irdl.type @complex {
    // CHECK:   irdl.parameters(elementType: AnyOf<f32, f64>)
    // CHECK: }
    irdl.type @complex {
      irdl.parameters(elementType: AnyOf<f32, f64>)
    }

    // CHECK: irdl.operation @norm {
    // CHECK:   irdl.constraint_vars(T: AnyOf<f32, f64>)
    // CHECK:   irdl.operands(c: @complex<?T>)
    // CHECK:   irdl.results(res: ?T)
    // CHECK: }
    irdl.operation @norm {
      irdl.constraint_vars(T: AnyOf<f32, f64>)
      irdl.operands(c: @complex<?T>)
      irdl.results(res: ?T)
    }

    // CHECK: irdl.operation @mul {
    // CHECK:   irdl.constraint_vars(T: AnyOf<f32, f64>)
    // CHECK:   irdl.operands(lhs: @complex<?T>, rhs: @complex<?T>)
    // CHECK:   irdl.results(res: @complex<?T>)
    // CHECK: }
    irdl.operation @mul {
      irdl.constraint_vars(T: AnyOf<f32, f64>)
      irdl.operands(lhs: @complex<?T>, rhs: @complex<?T>)
      irdl.results(res: @complex<?T>)
    }
  }
}
