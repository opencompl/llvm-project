// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics | FileCheck %s
// Verify that extensible dialects can register dynamic operations and types.

//===----------------------------------------------------------------------===//
// Dynamic type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @succeededDynamicTypeVerifier
func @succeededDynamicTypeVerifier() {
  // CHECK: %{{.*}} = "unregistered_op"() : () -> !test.singleton_dyntype
  // CHECK-NEXT: "unregistered_op"() : () -> !test.pair_dyntype<i32, f64>
  // CHECK_NEXT: %{{.*}} = "unregistered_op"() : () -> !test.pair_dyntype<!test.pair_dyntype<i32, f64>, !test.singleton_dyntype>
  "unregistered_op"() : () -> !test.singleton_dyntype
  "unregistered_op"() : () -> !test.pair_dyntype<i32, f64>
  "unregistered_op"() : () -> !test.pair_dyntype<!test.pair_dyntype<i32, f64>, !test.singleton_dyntype>
  return
}

// -----

func @failedDynamicTypeVerifier() {
  // expected-error@+1 {{expected 0 type arguments, but had 1}}
  "unregistered_op"() : () -> !test.singleton_dyntype<f64>
  return
}

// -----

func @failedDynamicTypeVerifier2() {
  // expected-error@+1 {{expected 2 type arguments, but had 1}}
  "unregistered_op"() : () -> !test.pair_dyntype<f64>
  return
}

// -----

// CHECK-LABEL: func @customTypeParserPrinter
func @customTypeParserPrinter() {
  // CHECK: "unregistered_op"() : () -> !test.custom_assembly_format_dyntype<f32:f64>
  "unregistered_op"() : () -> !test.custom_assembly_format_dyntype<f32 : f64>
  return
}

//===----------------------------------------------------------------------===//
// Dynamic op
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @succeededDynamicOpVerifier
func @succeededDynamicOpVerifier(%a: f32) {
  // CHECK: "test.generic_dynamic_op"() : () -> ()
  // CHECK-NEXT: %{{.*}} = "test.generic_dynamic_op"(%{{.*}}) : (f32) -> f64
  // CHECK-NEXT: %{{.*}}:2 = "test.one_operand_two_results"(%{{.*}}) : (f32) -> (f64, f64)
  "test.generic_dynamic_op"() : () -> ()
  "test.generic_dynamic_op"(%a) : (f32) -> f64
  "test.one_operand_two_results"(%a) : (f32) -> (f64, f64)
  return
}

// -----

func @failedDynamicOpVerifier() {
  // expected-error@+1 {{expected 1 operand, but had 0}}
  "test.one_operand_two_results"() : () -> (f64, f64)
  return
}

// -----

func @failedDynamicOpVerifier2(%a: f32) {
  // expected-error@+1 {{expected 2 results, but had 0}}
  "test.one_operand_two_results"(%a) : (f32) -> ()
  return
}

// -----

// CHECK-LABEL: func @customOpParserPrinter
func @customOpParserPrinter() {
  // CHECK: test.custom_parser_printer_dynamic_op custom_keyword
  test.custom_parser_printer_dynamic_op custom_keyword
  return
}

//===----------------------------------------------------------------------===//
// Dynamic trait
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @succeededDynamicTraitVerifier
func @succeededDynamicTraitVerifier() {
  // CHECK: "test.one_result_dynamic_op"() : () -> f32
  "test.one_result_dynamic_op"() : () -> f32
  return
}

// -----

func @failedDynamicTraitVerifier() {
  // expected-error@+1 {{requires one result}}
  "test.one_result_dynamic_op"() : () -> ()
  return
}

// -----

// CHECK-LABEL: func @succeededDynamicRuntimeTraitVerifier
func @succeededDynamicRuntimeTraitVerifier(%a: f32) {
  // CHECK: "test.two_operands_dynamic_op"(%{{.*}}, %{{.*}}) : (f32, f32) -> ()
  "test.two_operands_dynamic_op"(%a, %a) : (f32, f32) -> ()
  return
}

// -----

func @failedDynamicRuntimeTraitVerifier() {
  // expected-error@+1 {{requires two operands}}
  "test.two_operands_dynamic_op"() : () -> ()
  return
}
