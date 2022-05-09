func.func @assert_false() {
  %true = arith.constant false
  cf.assert %true, "Condition should have been satisfied."
  return
}

func.func @test_simplify_if(%N : index, %M : index, %K : index) {

  // assert condition on symbols.
  affine.if affine_set<()[N, M, K] : (N >= 0, 50 - N >= 0, M >= 0, 100 - M >= 0, K >= 0, 10 - K >= 0)>()[%N, %M, %K] {

    affine.for %i = 0 to %N {
      affine.for %j = 0 to %M {

        // assert condition on symbols.
        affine.if affine_set<(d0)[N] : (d0 >= 0, N - d0 >= 0, 25 - d0 >= 0)>(%i)[%N] {} 
        else {
          func.call @assert_false() : () -> ()
        }

        // assert condition on symbols.
        affine.if affine_set<(d0)[N] : (d0 >= 0, N - d0 >= 0, 120 - d0 >= 0)>(%j)[%M] {} 
        else {
          func.call @assert_false() : () -> ()
        }

        affine.for %k = 0 to %K {

          // assert condition on symbols.
          affine.if affine_set<(d0)[N] : (d0 >= 0, N - d0 >= 0, 9 - d0 >= 0)>(%k)[%K] {} 
          else {
            func.call @assert_false() : () -> ()
          }

        }

      }
    }

  } else {
    func.call @assert_false() : () -> ()
  }

  return
}
