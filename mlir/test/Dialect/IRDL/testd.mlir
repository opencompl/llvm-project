// RUN: mlir-irdl-opt %s --irdl-file=%S/testd.irdl.mlir -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Equality constraint
//===----------------------------------------------------------------------===//

func.func @succeededEqConstraint() {
  // CHECK: "testd.eq"() : () -> i32
  "testd.eq"() : () -> i32
  return
}

// -----

func.func @failedEqConstraint() {
  // expected-error@+1 {{expected 'i32' but got 'i64'}}
  "testd.eq"() : () -> i64
  return
}

// -----

//===----------------------------------------------------------------------===//
// AnyOf constraint
//===----------------------------------------------------------------------===//

func.func @succeededAnyOfConstraint() {
  // CHECK: "testd.anyof"() : () -> i32
  "testd.anyof"() : () -> i32
  // CHECK: "testd.anyof"() : () -> i64
  "testd.anyof"() : () -> i64
  return
}

// -----

func.func @failedAnyOfConstraint() {
  // expected-error@+1 {{'i1' does not satisfy the constraint}}
  "testd.anyof"() : () -> i1
  return
}

// -----

//===----------------------------------------------------------------------===//
// And constraint
//===----------------------------------------------------------------------===//

func.func @succeededAndConstraint() {
  // CHECK: "testd.and"() : () -> i64
  "testd.and"() : () -> i64
  return
}

// -----

func.func @failedAndConstraint1() {
  // expected-error@+1 {{'i1' does not satisfy the constraint}}
  "testd.and"() : () -> i1
  return
}

// -----

func.func @failedAndConstraint2() {
  // expected-error@+1 {{'i32' does not satisfy the constraint}}
  "testd.and"() : () -> i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// Any constraint
//===----------------------------------------------------------------------===//

func.func @succeededAnyConstraint() {
  // CHECK: "testd.any"() : () -> i32
  "testd.any"() : () -> i32
  // CHECK: "testd.any"() : () -> i64
  "testd.any"() : () -> i64
  return
}

// -----

//===----------------------------------------------------------------------===//
// Non-dynamic parameters constraint
//===----------------------------------------------------------------------===//

func.func @succeededParamsConstraint() {
  // CHECK: "testd.params"() : () -> complex<i32>
  "testd.params"() : () -> complex<i32>
  // CHECK: "testd.params"() : () -> complex<i64>
  "testd.params"() : () -> complex<i64>
  return
}

// -----

func.func @failedDynParamsConstraintBase() {
  // expected-error@+1 {{expected base type 'builtin.complex' but got type 'i32'}}
  "testd.params"() : () -> i32
  return
}

// -----

func.func @failedDynParamsConstraintParam() {
  // expected-error@+1 {{'i1' does not satisfy the constraint}}
  "testd.params"() : () -> complex<i1>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Dynamic base constraint
//===----------------------------------------------------------------------===//

func.func @succeededDynBaseConstraint() {
  // CHECK: "testd.dynbase"() : () -> !testd.parametric<i32>
  "testd.dynbase"() : () -> !testd.parametric<i32>
  // CHECK: "testd.dynbase"() : () -> !testd.parametric<i64>
  "testd.dynbase"() : () -> !testd.parametric<i64>
  // CHECK: "testd.dynbase"() : () -> !testd.parametric<!testd.parametric<i64>>
  "testd.dynbase"() : () -> !testd.parametric<!testd.parametric<i64>>
  return
}

// -----

func.func @failedDynBaseConstraint() {
  // expected-error@+1 {{expected base type 'testd.parametric' but got type 'i32'}}
  "testd.dynbase"() : () -> i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// Dynamic parameters constraint
//===----------------------------------------------------------------------===//

func.func @succeededDynParamsConstraint() {
  // CHECK: "testd.dynparams"() : () -> !testd.parametric<i32>
  "testd.dynparams"() : () -> !testd.parametric<i32>
  // CHECK: "testd.dynparams"() : () -> !testd.parametric<i64>
  "testd.dynparams"() : () -> !testd.parametric<i64>
  return
}

// -----

func.func @failedDynParamsConstraintBase() {
  // expected-error@+1 {{expected base type 'testd.parametric' but got type 'i32'}}
  "testd.dynparams"() : () -> i32
  return
}

// -----

func.func @failedDynParamsConstraintParam() {
  // expected-error@+1 {{'i1' does not satisfy the constraint}}
  "testd.dynparams"() : () -> !testd.parametric<i1>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Constraint variables
//===----------------------------------------------------------------------===//

func.func @succeededConstraintVars() {
  // CHECK: "testd.constraint_vars"() : () -> (i32, i32)
  "testd.constraint_vars"() : () -> (i32, i32)
  return
}

// -----

func.func @succeededConstraintVars2() {
  // CHECK: "testd.constraint_vars"() : () -> (i64, i64)
  "testd.constraint_vars"() : () -> (i64, i64)
  return
}

// -----

func.func @failedConstraintVars() {
  // expected-error@+1 {{expected 'i64' but got 'i32'}}
  "testd.constraint_vars"() : () -> (i64, i32)
  return
}
