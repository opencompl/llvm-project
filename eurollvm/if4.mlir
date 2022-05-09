func.func @assert_false() {
  %true = arith.constant false
  cf.assert %true, "Condition should have been satisfied."
  return
}

func.func @simplify_if(%N : index, %M : index) {

  affine.for %i = 0 to %N {
    affine.for %j = affine_map<(i) -> (i + 1)>(%i) to %M {

      // assert on symbols.
      affine.if affine_set<()[N, M] : (N >= 0, M >= 0, 100 - N >= 0, 100 - M >= 0, N - M >= 0)>()[%N, %M] {

        affine.if affine_set<()[N, M] : (50 - N >= 0, 100 - M >= 0)>()[%N, %M] {
          "something.thencall"() : () -> ()
        } else {
          "something.elsecall"() : () -> ()
        }

      } else {
        func.call @assert_false() : () -> ()
      }

    }
  }


  return
}
