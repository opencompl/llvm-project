#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
// #include "../unittests/Analysis/Presburger/Utils.h"
#include "mlir/IR/MLIRContext.h"
#include "../unittests/Dialect/Affine/Analysis/AffineStructuresParser.h"


#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <fstream>


#include "benchmark/benchmark.h"

// #define N (1024 * 16)

using namespace mlir;

/// Parses a IntegerPolyhedron from a StringRef. It is expected that the
/// string represents a valid IntegerSet, otherwise it will violate a gtest
/// assertion.
presburger::IntegerPolyhedron parsePoly(llvm::StringRef str) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  FailureOr<presburger::IntegerPolyhedron> poly = parseIntegerSetToFAC(str, &context);
  // EXPECT_TRUE(succeeded(poly));
  return *poly;
}

// IntegerPolyhedron parsePoly(llvm::StringRef str) {
//   MLIRContext context(MLIRContext::Threading::DISABLED);
//   FailureOr<IntegerPolyhedron> poly = parseIntegerSetToFAC(str, &context);
//   // EXPECT_TRUE(succeeded(poly));
//   return *poly;
// }

static void BM_PresburgerSetUnion(benchmark::State& state) {
  std::ifstream file("../mlir/benchmark/PresburgerSetTestCase");
  std::string line;
  std::getline(file, line);
  presburger::PresburgerSet setA{parsePoly(line)};
  std::getline(file, line);
  presburger::PresburgerSet setB{parsePoly(line)};
  std::getline(file, line);
  presburger::PresburgerSet setC{parsePoly(line)};
  
  presburger::PresburgerSet res = setA.unionSet(setB);
  // EXPECT_TRUE(res.isEqual(setC));
}
BENCHMARK(BM_PresburgerSetUnion);

static void BM_PresburgerSetIntersect(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
BENCHMARK(BM_PresburgerSetIntersect);

static void BM_PresburgerSetSubtract(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
BENCHMARK(BM_PresburgerSetSubtract);

static void BM_PresburgerSetComplement(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
BENCHMARK(BM_PresburgerSetComplement);

static void BM_PresburgerSetIsEqual(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
BENCHMARK(BM_PresburgerSetIsEqual);

static void BM_PresburgerSetIsEmpty(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
BENCHMARK(BM_PresburgerSetIsEmpty);



// BENCHMARK_MAIN();

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  benchmark::RunSpecifiedBenchmarks();
}