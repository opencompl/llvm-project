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

#include <iostream>
static void BM_PresburgerSetUnion(benchmark::State& state) {
  for (auto _ : state) {}
  // #pragma nounroll

  std::ifstream file("../mlir/benchmark/PresburgerSetUnion");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);    
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setC{parsePoly(line)};
    
    presburger::PresburgerSet res = setA.unionSet(setB);
    assert(res.isEqual(setC));
  }

  file.close();
  // state.SkipWithError("error message");
}
BENCHMARK(BM_PresburgerSetUnion);

static void BM_PresburgerSetIntersect(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
  
  std::ifstream file("../mlir/benchmark/PresburgerSetIntersect");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);    
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setC{parsePoly(line)};
    
    presburger::PresburgerSet res = setA.intersect(setB);
    assert(res.isEqual(setC));
  }

  file.close();
}
BENCHMARK(BM_PresburgerSetIntersect);

static void BM_PresburgerSetSubtract(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
  
  std::ifstream file("../mlir/benchmark/PresburgerSetSubtract");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);    
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setC{parsePoly(line)};
    
    presburger::PresburgerSet res = setA.subtract(setB);
    assert(res.isEqual(setC));
  }

  file.close();
}
BENCHMARK(BM_PresburgerSetSubtract);

static void BM_PresburgerSetComplement(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;

  std::ifstream file("../mlir/benchmark/PresburgerSetComplement");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);    
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};
    
    presburger::PresburgerSet res = setA.complement();
    assert(res.isEqual(setB));
  }

  file.close();

}
BENCHMARK(BM_PresburgerSetComplement);

static void BM_PresburgerSetIsEqual(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;

  std::ifstream file("../mlir/benchmark/PresburgerSetEqual");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);

  for (int i = 0; i < num; i++) {
    bool result;
    std::getline(file, line);
    std::istringstream(line) >> result;
    std::getline(file, line);
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};

    if (result)
      assert(setA.isEqual(setB));
    else
      assert(!setA.isEqual(setB)); 
  }

  file.close();
}
BENCHMARK(BM_PresburgerSetIsEqual);

static void BM_PresburgerSetIsEmpty(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;

  std::ifstream file("../mlir/benchmark/PresburgerSetEmpty");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);

  for (int i = 0; i < num; i++) {
    bool result;
    std::getline(file, line);
    std::istringstream(line) >> result;
    std::getline(file, line);
    presburger::PresburgerSet setA{parsePoly(line)};

    if (result)
      assert(setA.isIntegerEmpty());
    else
      assert(!setA.isIntegerEmpty()); 
  }

  file.close();
}
BENCHMARK(BM_PresburgerSetIsEmpty);

// BENCHMARK_MAIN();

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  benchmark::RunSpecifiedBenchmarks();
}