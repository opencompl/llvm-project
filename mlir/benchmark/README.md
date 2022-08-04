# README

This is the Google Benchmark Test for 6 basic Presburger set operations.

To execute the following commands, enter the `build` directory under llvm-project.

## Execution

The test cases to be used are compressed in `PresburgerSets.tar.gz`. And further operations are shown as below.

``` bash
# Extract test cases
tar -xzf ../mlir/benchmark/testcases.tar.gz -C ../mlir/benchmark

# Compile
ninja PresburgerSetBenchmark

# Run the performance test
./tools/mlir/benchmark/PresburgerSetBenchmark
./tools/mlir/benchmark/PresburgerSetBenchmark --benchmark_repetitions=3 > ../mlir/benchmark/fpl
```

The extracted files should be deleted when switch branches.

```
rm ../mlir/benchmark/Presburger*
```
