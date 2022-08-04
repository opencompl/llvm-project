add_benchmark(PresburgerSetBenchmark set.bench.cpp   ../unittests/Dialect/Affine/Analysis/AffineStructuresParser.cpp
)

target_link_libraries(PresburgerSetBenchmark
  PRIVATE
  MLIRPresburger
  LLVMSupport
  MLIRAffineAnalysis
  MLIRParser

  )
