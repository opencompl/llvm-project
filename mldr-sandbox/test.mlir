func.func @test (%arg0: i32) -> i32 {
  %0 = arith.constant 2 : i32
  %1 = arith.muli %0, %arg0 : i32 
  return %1 : i32
}
