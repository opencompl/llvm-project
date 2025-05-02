func.func @test (%arg0: i32) -> i32 {
  %0 = arith.constant 2 : i32
  %1 = arith.muli %0, %arg0 : i32 
  %2 = arith.constant 2 : i32
  %3 = arith.muli %1, %2 : i32 
  return %3 : i32
}
