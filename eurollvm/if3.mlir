func.func @test_simplify_if(%N : index, %M : index, %K : index) {
  affine.if affine_set<()[N, M] : (N - M >= 0)>()[%N, %M] {

    affine.for %i = 0 to %N {

      affine.if affine_set<()[M] : (M >= 0)>()[%M] {

      }

      affine.if affine_set<(i)[N, M, K] : (N - i >= 0, i - M >= 0, i - K >= 0)>(%i)[%N, %M, %K] {

      } else {

        affine.if affine_set<(i)[K] : (i - K >= 0)>(%i)[%K] {

        }

      }

    }

  }

  return
}
