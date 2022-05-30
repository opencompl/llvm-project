#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
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
  return *poly;
}

/// Parses a list of comma separated IntegerSets to IntegerPolyhedron and
/// combine them into a PresburgerSet by using the union operation. It is
/// expected that the string has valid comma separated IntegerSet constraints
/// and that all of them have the same number of dimensions as is specified by
/// the numDims argument.
presburger::PresburgerSet parsePresburgerSet(StringRef str) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  FailureOr<SmallVector<FlatAffineValueConstraints, 4>> facs =
      parseMultipleIntegerSetsToFAC(str, &context);
  SmallVector<presburger::IntegerPolyhedron, 4> ips;
  for (auto fac : facs.getValue())
    ips.push_back(presburger::IntegerPolyhedron(fac));

  presburger::PresburgerSet set = presburger::PresburgerSet(ips.front());
  for (int i = 1, m = facs.getValue().size(); i < m; i++)
    set.unionInPlace(ips[i]);
  return set;
}

static void BM_PresburgerSetUnion(benchmark::State& state) {
  std::ifstream file("../mlir/benchmark/PresburgerSetUnion");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  std::vector<presburger::PresburgerSet> setsA;
  std::vector<presburger::PresburgerSet> setsB;
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};
    setsA.push_back(setA);
    setsB.push_back(setB);
  }

  file.close();

  for (auto _ : state) {
    for (int i = 0; i < num; i++)
      benchmark::DoNotOptimize(setsA[i].unionSet(setsB[i]));
  }
}
BENCHMARK(BM_PresburgerSetUnion);

static void BM_PresburgerSetIntersect(benchmark::State& state) {
  std::ifstream file("../mlir/benchmark/PresburgerSetIntersect");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  std::vector<presburger::PresburgerSet> setsA;
  std::vector<presburger::PresburgerSet> setsB;
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};
    setsA.push_back(setA);
    setsB.push_back(setB);
  }

  file.close();

  for (auto _ : state) {   
    for (int i = 0; i < num; i++)
      benchmark::DoNotOptimize(setsA[i].intersect(setsB[i]));
  }
}
BENCHMARK(BM_PresburgerSetIntersect);

static void BM_PresburgerSetSubtract(benchmark::State& state) {
  std::ifstream file("../mlir/benchmark/PresburgerSetSubtract");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  std::vector<presburger::PresburgerSet> setsA;
  std::vector<presburger::PresburgerSet> setsB;
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);    
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};
    setsA.push_back(setA);
    setsB.push_back(setB);
  }

  file.close();

  for (auto _ : state) {
    for (int i = 0; i < num; i++) {
      benchmark::DoNotOptimize(setsA[i].subtract(setsB[i]));
    }
  }
}
BENCHMARK(BM_PresburgerSetSubtract);

static void BM_PresburgerSetComplement(benchmark::State& state) {
  std::ifstream file("../mlir/benchmark/PresburgerSetComplement");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  std::vector<presburger::PresburgerSet> setsA;
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);    
    presburger::PresburgerSet setA{parsePoly(line)};
    setsA.push_back(setA);
  }

  file.close();

  for (auto _ : state) {
    for (int i = 0; i < num; i++)
      benchmark::DoNotOptimize(setsA[i].complement());
  }
}
BENCHMARK(BM_PresburgerSetComplement);

static void BM_PresburgerSetIsEqual(benchmark::State& state) {
  std::ifstream file("../mlir/benchmark/PresburgerSetEqual");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  std::vector<presburger::PresburgerSet> setsA;
  std::vector<presburger::PresburgerSet> setsB;
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);    
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};    
    setsA.push_back(setA);
    setsB.push_back(setB);
  }

  file.close();

  for (auto _ : state) {
    for (int i = 0; i < num; i++)
      benchmark::DoNotOptimize(setsA[i].isEqual(setsB[i]));
  }
}
BENCHMARK(BM_PresburgerSetIsEqual);

static void BM_PresburgerSetIsEmpty(benchmark::State& state) {
  std::ifstream file("../mlir/benchmark/PresburgerSetEmpty");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  std::vector<presburger::PresburgerSet> setsA;
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);    
    presburger::PresburgerSet setA = parsePresburgerSet(line);
    std::getline(file, line);    
    setsA.push_back(setA);
  }

  file.close();

  for (auto _ : state) {
    for (int i = 0; i < num; i++)
      benchmark::DoNotOptimize(setsA[i].isIntegerEmpty());
  }
}
BENCHMARK(BM_PresburgerSetIsEmpty);

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  benchmark::RunSpecifiedBenchmarks();
}