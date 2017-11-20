//#include <benchmark/benchmark.h>
//#include "dct.h"
//
//static void BM_GPU_BATCH_DST(benchmark::State& state) {
//	int batch = state.range(0);
//	int16_t *src = generateArrayBatch(4, batch);
//	int16_t *dst = new int16_t[4 * 4 * batch];
//	cudaAlloc(4, batch);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(dst4_c(src, dst, GPUAtomic, batch));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_BATCH_DST)->ArgName("M")->Arg(350)->UseManualTime();
//
//static void BM_GPU_DST(benchmark::State& state) {
//
//	int16_t *src = generateArray(4);
//	int16_t *dst = new int16_t[4 * 4];
//	PTYPE type = static_cast<PTYPE>(state.range(0));
//	cudaAlloc(4);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(dst4_c(src, dst, type));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_DST)->Arg(6)->UseManualTime();
//
//static void BM_GPU_BATCH_DCT4(benchmark::State& state) {
//	int batch = state.range(0);
//	int16_t *src = generateArrayBatch(4, batch);
//	int16_t *dst = new int16_t[4 * 4 * batch];
//	cudaAlloc(4, batch);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(dct4_c(src, dst, GPUAtomic, batch));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_BATCH_DCT4)->ArgName("M")->Arg(350)->UseManualTime();
//
//static void BM_GPU_DCT4(benchmark::State& state) {
//
//	int16_t *src = generateArray(4);
//	int16_t *dst = new int16_t[4 * 4];
//	PTYPE type = static_cast<PTYPE>(state.range(0));
//	cudaAlloc(4);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(dct4_c(src, dst, type));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_DCT4)->Arg(6)->UseManualTime();
//
////->ArgNames({ "Plain", "Shared Memory", "Minimum Multiplications", "One Step", "Atomic" })
//
//static void BM_GPU_BATCH_DCT8(benchmark::State& state) {
//	int batch = state.range(0);
//	int16_t *src = generateArrayBatch(8, batch);
//	int16_t *dst = new int16_t[8 * 8 * batch];
//	cudaAlloc(8, batch);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(dct8_c(src, dst, GPUAtomic, batch));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_BATCH_DCT8)->ArgName("M")->Arg(250)->UseManualTime();
//
//static void BM_GPU_DCT8(benchmark::State& state) {
//
//	int16_t *src = generateArray(8);
//	int16_t *dst = new int16_t[8 * 8];
//	PTYPE type = static_cast<PTYPE>(state.range(0));
//	cudaAlloc(8);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(dct8_c(src, dst, type));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_DCT8)->Arg(6)->UseManualTime();
//
//
//static void BM_GPU_BATCH_DCT16(benchmark::State& state) {
//	int batch = state.range(0);
//	int16_t *src = generateArrayBatch(16, batch);
//	int16_t *dst = new int16_t[16 * 16 * batch];
//	cudaAlloc(16, batch);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(dct16_c(src, dst, GPUAtomic, batch));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_BATCH_DCT16)->ArgName("M")->Arg(100)->UseManualTime();
//
//static void BM_GPU_DCT16(benchmark::State& state) {
//
//	int16_t *src = generateArray(16);
//	int16_t *dst = new int16_t[16 * 16];
//	PTYPE type = static_cast<PTYPE>(state.range(0));
//	cudaAlloc(16);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(dct16_c(src, dst, type));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_DCT16)->Arg(6)->UseManualTime();
//
//
//static void BM_GPU_BATCH_DCT32(benchmark::State& state) {
//	int batch = state.range(0);
//	int16_t *src = generateArrayBatch(32, batch);
//	int16_t *dst = new int16_t[32 * 32 * batch];
//	cudaAlloc(32, batch);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(dct32_c(src, dst, GPUAtomic, batch));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_BATCH_DCT32)->ArgName("M")->Arg(30)->UseManualTime();
//
//static void BM_GPU_DCT32(benchmark::State& state) {
//
//	int16_t *src = generateArray(32);
//	int16_t *dst = new int16_t[32 * 32];
//	PTYPE type = static_cast<PTYPE>(state.range(0));
//	cudaAlloc(32);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(dct32_c(src, dst, type));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_DCT32)->Arg(6)->UseManualTime();
//
//static void BM_GPU_BATCH_IDST(benchmark::State& state) {
//	int batch = state.range(0);
//	int16_t *src = generateArrayBatch(4, batch);
//	int16_t *dst = new int16_t[4 * 4 * batch];
//	cudaAlloc(4, batch);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(idst4_c(src, dst, GPUAtomic, batch));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_BATCH_IDST)->ArgName("M")->Arg(350)->UseManualTime();
//
//static void BM_GPU_IDST(benchmark::State& state) {
//
//	int16_t *src = generateArray(4);
//	int16_t *dst = new int16_t[4 * 4];
//	PTYPE type = static_cast<PTYPE>(state.range(0));
//	cudaAlloc(4);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(idst4_c(src, dst, type));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_IDST)->Arg(6)->UseManualTime();
//
//
//static void BM_GPU_BATCH_IDCT4(benchmark::State& state) {
//	int batch = state.range(0);
//	int16_t *src = generateArrayBatch(4, batch);
//	int16_t *dst = new int16_t[4 * 4 * batch];
//	cudaAlloc(4, batch);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(idct4_c(src, dst, GPUAtomic, batch));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_BATCH_IDCT4)->ArgName("M")->Arg(350)->UseManualTime();
//
//static void BM_GPU_IDCT4(benchmark::State& state) {
//
//	int16_t *src = generateArray(4);
//	int16_t *dst = new int16_t[4 * 4];
//	PTYPE type = static_cast<PTYPE>(state.range(0));
//	cudaAlloc(4);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(idct4_c(src, dst, type));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_IDCT4)->Arg(6)->UseManualTime();
//
//
//static void BM_GPU_BATCH_IDCT8(benchmark::State& state) {
//	int batch = state.range(0);
//	int16_t *src = generateArrayBatch(8, batch);
//	int16_t *dst = new int16_t[8 * 8 * batch];
//	cudaAlloc(8, batch);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(idct8_c(src, dst, GPUAtomic, batch));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_BATCH_IDCT8)->ArgName("M")->Arg(150)->UseManualTime();
//
//static void BM_GPU_IDCT8(benchmark::State& state) {
//
//	int16_t *src = generateArray(8);
//	int16_t *dst = new int16_t[8 * 8];
//	PTYPE type = static_cast<PTYPE>(state.range(0));
//	cudaAlloc(8);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(idct8_c(src, dst, type));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_IDCT8)->Arg(6)->UseManualTime();
//
//
//static void BM_GPU_BATCH_IDCT16(benchmark::State& state) {
//	int batch = state.range(0);
//	int16_t *src = generateArrayBatch(16, batch);
//	int16_t *dst = new int16_t[16 * 16 * batch];
//	cudaAlloc(16, batch);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(idct16_c(src, dst, GPUAtomic, batch));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_BATCH_IDCT16)->ArgName("M")->Arg(100)->UseManualTime();
//
//static void BM_GPU_IDCT16(benchmark::State& state) {
//
//	int16_t *src = generateArray(16);
//	int16_t *dst = new int16_t[16 * 16];
//	PTYPE type = static_cast<PTYPE>(state.range(0));
//	cudaAlloc(16);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(idct16_c(src, dst, type));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_IDCT16)->Arg(6)->UseManualTime();
//
//
//static void BM_GPU_BATCH_IDCT32(benchmark::State& state) {
//	int batch = state.range(0);
//	int16_t *src = generateArrayBatch(32, batch);
//	int16_t *dst = new int16_t[32 * 32 * batch];
//	cudaAlloc(32, batch);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(idct32_c(src, dst, GPUAtomic, batch));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_BATCH_IDCT32)->ArgName("M")->Arg(100)->UseManualTime();
//
//static void BM_GPU_IDCT32(benchmark::State& state) {
//
//	int16_t *src = generateArray(32);
//	int16_t *dst = new int16_t[32 * 32];
//	PTYPE type = static_cast<PTYPE>(state.range(0));
//	cudaAlloc(32);
//	for (auto _ : state){
//		auto start = std::chrono::high_resolution_clock::now();
//		benchmark::DoNotOptimize(idct32_c(src, dst, type));
//		benchmark::ClobberMemory();
//		auto end = std::chrono::high_resolution_clock::now();
//		auto elapsed_seconds =
//			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//		state.SetIterationTime(elapsed_seconds.count());
//	}
//	cudaDestroy();
//}
//BENCHMARK(BM_GPU_IDCT32)->Arg(6)->UseManualTime();
//
//BENCHMARK_MAIN();