module {
  func.func @foo(%1 : i32) -> i32 {
    %2 = arith.constant 3 : i32
    %3 = arith.ori %1, %2 : i32
    func.return %3 : i32
  }
}