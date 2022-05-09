func.func @test_simplify_if() {
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to affine_map<(d0) -> (d0)>(%arg0) {
      affine.if affine_set<(d0, d1) : (10 - d0 >= 0, d0 - 5 >= 0, 50 - d1 >= 0, d1 >= 0)>(%arg0, %arg1) {}
      else {
        affine.if affine_set<(d0, d1) : (10 - d0 >= 0, d0 - 5 >= 0, 50 - d1 >= 0, d1 >= 0)>(%arg0, %arg1) {}
      }
    }
  }
  return
}
