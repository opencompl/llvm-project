module {
  func.func @foo() -> i32 {
    %1 = arith.constant 5 : i32
    func.return %1 : i32
  }
}