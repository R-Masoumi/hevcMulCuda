//#include "runner.h"
//
//int main(){
//	int16_t *src, *dst, *dst2;
//	const int m = 32;
//	int batch = 2;
//	src = generateArrayBatch(m,batch);
//	dst = new int16_t[m * m*batch];
//	dst2 = new int16_t[m * m*batch];
//	cudaAlloc(m,batch);
//	print_matrix(src,m,batch);
//	testRun(DCT32, GPUAtomic, src, dst, batch);
//	testRun(IDCT32, GPUAtomic, dst, dst2, batch);
//	const int n = 32 ;
//	delete(src);
//	delete(dst);
//	delete(dst2);
//	src = generateArrayBatch(n, batch);
//	dst = new int16_t[n * n* batch];
//	dst2 = new int16_t[n * n* batch];
//	print_matrix(src, n, batch);
//	testRun(DCT32, GPUAtomic, src, dst, batch);
//	testRun(IDCT32, GPUAtomic, dst, dst2,batch);
//	cudaDestroy();
//}