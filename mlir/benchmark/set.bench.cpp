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
  std::ifstream file("../mlir/benchmark/PresburgerSetUnion");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  std::vector<presburger::PresburgerSet> setsA;
  std::vector<presburger::PresburgerSet> setsB;
  // std::vector<presburger::PresburgerSet> res;
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};
    std::getline(file, line);
    // presburger::PresburgerSet setC{parsePoly(line)};

    setsA.push_back(setA);
    setsB.push_back(setB);
    // sets.push_back(setC);
  }

  file.close();

  for (auto _ : state) {
    for (int i = 0; i < num; i++) {
      setsA[i].unionSet(setsB[i]);
      // sets[i * 3].unionSet(sets[i * 3 + 1]);
      // res.push_back(sets[i * 3].unionSet(sets[i * 3 + 1]));
    }
  }

  // for (int i = 0; i < num; i++) {
  //   assert(res[i].isEqual(sets[i * 3 + 2]));
  // }

  // llvm::errs() << "test";
  // benchmark::DoNotOptimize(res);
  // state.SkipWithError("error message");
}
BENCHMARK(BM_PresburgerSetUnion);

static void BM_PresburgerSetIntersect(benchmark::State& state) {
  std::ifstream file("../mlir/benchmark/PresburgerSetIntersect");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  std::vector<presburger::PresburgerSet> setsA;
  std::vector<presburger::PresburgerSet> setsB;
  // std::vector<presburger::PresburgerSet> res;
  
  for (int i = 0; i < num; i++) {
    std::getline(file, line);
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};
    std::getline(file, line);
    // presburger::PresburgerSet setC{parsePoly(line)};

    setsA.push_back(setA);
    setsB.push_back(setB);
    // sets.push_back(setC);
  }

  file.close();

  for (auto _ : state) {   
    for (int i = 0; i < num; i++) {
      setsA[i].intersect(setsB[i]);
    }
  }

  // assert(res.isEqual(setC));
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
    std::getline(file, line);
    // presburger::PresburgerSet setC{parsePoly(line)};
    
    setsA.push_back(setA);
    setsB.push_back(setB);
  }

  file.close();

  for (auto _ : state) {
    for (int i = 0; i < num; i++) {
      setsA[i].subtract(setsB[i]);
    }
  }

  // assert(res.isEqual(setC));

}
BENCHMARK(BM_PresburgerSetSubtract);

static void BM_PresburgerSetComplement(benchmark::State& state) {
  std::ifstream file("../mlir/benchmark/PresburgerSetComplement");
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
      setsA[i].complement();
    }
  }

  // for (int i = 0; i < num; i++) {
  //   assert(res.isEqual(setB));
  // }
}
BENCHMARK(BM_PresburgerSetComplement);

static void BM_PresburgerSetIsEqual(benchmark::State& state) {
  std::ifstream file("../mlir/benchmark/PresburgerSetEqual");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  std::vector<presburger::PresburgerSet> setsA;
  std::vector<presburger::PresburgerSet> setsB;
  std::vector<bool> results;
  
  for (int i = 0; i < num; i++) {
    bool result;
    std::getline(file, line);
    std::istringstream(line) >> result;
    std::getline(file, line);    
    presburger::PresburgerSet setA{parsePoly(line)};
    std::getline(file, line);
    presburger::PresburgerSet setB{parsePoly(line)};
    
    results.push_back(result);
    setsA.push_back(setA);
    setsB.push_back(setB);
  }

  file.close();

  for (auto _ : state) {
    for (int i = 0; i < num; i++) {
      if (results[i])
        assert(setsA[i].isEqual(setsB[i]));
      else
        assert(!setsA[i].isEqual(setsB[i])); 
    }
  }
}
BENCHMARK(BM_PresburgerSetIsEqual);

static void BM_PresburgerSetIsEmpty(benchmark::State& state) {
  std::ifstream file("../mlir/benchmark/PresburgerSetEmpty");
  std::string line;
  std::getline(file, line);
  int num = stoi(line);
  std::vector<presburger::PresburgerSet> setsA;
  std::vector<bool> results;
  
  for (int i = 0; i < num; i++) {
    bool result;
    std::getline(file, line);
    std::istringstream(line) >> result;
    std::getline(file, line);    
    presburger::PresburgerSet setA{parsePoly(line)};
    
    results.push_back(result);
    setsA.push_back(setA);
  }

  file.close();

  for (auto _ : state) {
    for (int i = 0; i < num; i++) {
      if (results[i])
        assert(setsA[i].isIntegerEmpty());
      else
        assert(!setsA[i].isIntegerEmpty()); 
    }
  }
}
BENCHMARK(BM_PresburgerSetIsEmpty);

// BENCHMARK_MAIN();

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  benchmark::RunSpecifiedBenchmarks();
}