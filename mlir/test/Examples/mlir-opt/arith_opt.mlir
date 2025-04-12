func.func @optme(%arg0 : i32) -> i32 {
  %0 = arith.constant 2 : i32
  %1 = arith.muli %arg0, %0 : i32 
  %2 = arith.muli %1, %0 : i32 
  return %2 : i32
}