module {
func.func @foo() -> i1 {
  %x = arith.constant 1 : i1
  func.return %x : i1
}
}