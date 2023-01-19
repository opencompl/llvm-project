module {
func.func @foo(%arg0: i1, %arg1: f32) -> f32{
  %res = scf.if %arg0 -> f32 {
    scf.yield %arg1 : f32
  } else {
    %y = "dont.hoist"() : () -> f32
    scf.yield %y : f32
  }
  return %res : f32
}
}