add_benchmark(PresburgerSetBenchmark set.bench.cpp)

target_link_libraries(PresburgerSetBenchmark
  PRIVATE
  MLIRPresburger
  LLVMSupport
  MLIRAffineAnalysis
  MLIRParser

  )
