#include "gdct.cuh"

template<typename T>
__device__ inline T g_x265_clip3(T minVal, T maxVal, T a) { return g_x265_min(g_x265_max(minVal, a), maxVal); }
template<typename T>
__device__ inline T g_x265_min(T a, T b) { return a < b ? a : b; }
template<typename T>
__device__ inline T g_x265_max(T a, T b) { return a > b ? a : b; }


int16_t *src, *src2, *dst;
void cudaAlloc(int n){
	cudaMalloc(&dst, n * n * sizeof(int16_t));
	cudaMalloc(&src, n * n * sizeof(int16_t));
	cudaMalloc(&src2, n * n * sizeof(int16_t));
}

void cudaAlloc(int n,int m){
	cudaMalloc(&dst, m* n * n * sizeof(int16_t));
	cudaMalloc(&src, m* n * n * sizeof(int16_t));
	cudaMalloc(&src2, m* n * n * sizeof(int16_t));
}

void cudaDestroy(){
	cudaFree(src);
	cudaFree(src2);
	cudaFree(dst);
}

__global__ void fastForwardDst(const int16_t* block, int16_t* coeff, int shift)  // input block, output coeff
{
	int c[4];
	int rnd_factor = 1 << (shift - 1);
	int i = threadIdx.x;
	// Intermediate Variables
	c[0] = block[4 * i + 0] + block[4 * i + 3];
	c[1] = block[4 * i + 1] + block[4 * i + 3];
	c[2] = block[4 * i + 0] - block[4 * i + 1];
	c[3] = 74 * block[4 * i + 2];

	coeff[i] = (int16_t)((29 * c[0] + 55 * c[1] + c[3] + rnd_factor) >> shift);
	coeff[4 + i] = (int16_t)((74 * (block[4 * i + 0] + block[4 * i + 1] - block[4 * i + 3]) + rnd_factor) >> shift);
	coeff[8 + i] = (int16_t)((29 * c[2] + 55 * c[0] - c[3] + rnd_factor) >> shift);
	coeff[12 + i] = (int16_t)((55 * c[2] - 29 * c[1] + c[3] + rnd_factor) >> shift);
}

__global__ void inversedst(const int16_t* tmp, int16_t* block, int shift)  // input tmp, output block
{
	int c[4];
	int rnd_factor = 1 << (shift - 1);
	int i = threadIdx.x;//max 4

	// Intermediate Variables
	c[0] = tmp[i] + tmp[8 + i];
	c[1] = tmp[8 + i] + tmp[12 + i];
	c[2] = tmp[i] - tmp[12 + i];
	c[3] = 74 * tmp[4 + i];

	block[4 * i + 0] = (int16_t)g_x265_clip3(-32768, 32767, (29 * c[0] + 55 * c[1] + c[3] + rnd_factor) >> shift);
	block[4 * i + 1] = (int16_t)g_x265_clip3(-32768, 32767, (55 * c[2] - 29 * c[1] + c[3] + rnd_factor) >> shift);
	block[4 * i + 2] = (int16_t)g_x265_clip3(-32768, 32767, (74 * (tmp[i] - tmp[8 + i] + tmp[12 + i]) + rnd_factor) >> shift);
	block[4 * i + 3] = (int16_t)g_x265_clip3(-32768, 32767, (55 * c[0] + 29 * c[2] - c[3] + rnd_factor) >> shift);
}

__global__ void partialButterfly16(const int16_t* src, int16_t* dst, int shift, int line)
{
	__shared__ int E[8], O[8];
	__shared__ int EE[4], EO[4];
	int EEE[2], EEO[2];
	int add = 1 << (shift - 1);
	int j = blockIdx.x;//max 16
	int k = threadIdx.x;//max 16
	src += 16 * j;
	dst += j;

	/* E and O */
	if (k < 8)
	{
		E[k] = src[k] + src[15 - k];
		O[k] = src[k] - src[15 - k];
	}

	__syncthreads();

	/* EE and EO */
	if (k < 4)
	{
		EE[k] = E[k] + E[7 - k];
		EO[k] = E[k] - E[7 - k];
	}
	__syncthreads();
	if (k == 4)//unused thread
	{
		/* EEE and EEO */
		EEE[0] = EE[0] + EE[3];
		EEO[0] = EE[0] - EE[3];
		EEE[1] = EE[1] + EE[2];
		EEO[1] = EE[1] - EE[2];

		dst[0] = (int16_t)((t16[0][0] * EEE[0] + t16[0][1] * EEE[1] + add) >> shift);
		dst[8 * line] = (int16_t)((t16[8][0] * EEE[0] + t16[8][1] * EEE[1] + add) >> shift);
		dst[4 * line] = (int16_t)((t16[4][0] * EEO[0] + t16[4][1] * EEO[1] + add) >> shift);
		dst[12 * line] = (int16_t)((t16[12][0] * EEO[0] + t16[12][1] * EEO[1] + add) >> shift);
	}
	else if (k % 4 == 2)//2 6 10 14
	{
		dst[k * line] = (int16_t)((t16[k][0] * EO[0] + t16[k][1] * EO[1] + t16[k][2] * EO[2] +
			t16[k][3] * EO[3] + add) >> shift);
	}

	else if (k % 2 == 1)
	{
		dst[k * line] = (int16_t)((t16[k][0] * O[0] + t16[k][1] * O[1] + t16[k][2] * O[2] + t16[k][3] * O[3] +
			t16[k][4] * O[4] + t16[k][5] * O[5] + t16[k][6] * O[6] + t16[k][7] * O[7] +
			add) >> shift);
	}
}

__global__ void partialButterfly32(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j = blockIdx.x;// max 32
	int k = threadIdx.x;//max 32
	__shared__ int E[16], O[16];
	__shared__ int EE[8], EO[8];
	__shared__ int EEE[4], EEO[4];
	int EEEE[2], EEEO[2];
	int add = 1 << (shift - 1);

	src += 32 * j;
	dst += j;

	/* E and O*/
	if (k < 16)
	{
		E[k] = src[k] + src[31 - k];
		O[k] = src[k] - src[31 - k];
	}
	__syncthreads();

	/* EE and EO */
	if (k < 8)
	{
		EE[k] = E[k] + E[15 - k];
		EO[k] = E[k] - E[15 - k];
	}
	__syncthreads();

	/* EEE and EEO */
	if (k < 4)
	{
		EEE[k] = EE[k] + EE[7 - k];
		EEO[k] = EE[k] - EE[7 - k];
	}
	__syncthreads();



	/* EEEE and EEEO */
	EEEE[0] = EEE[0] + EEE[3];
	EEEO[0] = EEE[0] - EEE[3];
	EEEE[1] = EEE[1] + EEE[2];
	EEEO[1] = EEE[1] - EEE[2];

	if (k == 1 || k == 3){
		dst[(k - 1) * 8 * line] = (int16_t)((t32[(k - 1) * 8][0] * EEEE[0] + t32[(k - 1) * 8][1] * EEEE[1] + add) >> shift);
		dst[k * 8 * line] = (int16_t)((t32[k * 8][0] * EEEO[0] + t32[k * 8][1] * EEEO[1] + add) >> shift);
	}
	if (k % 8 == 4)//4 12 20 28
	{
		dst[k * line] = (int16_t)((t32[k][0] * EEO[0] + t32[k][1] * EEO[1] + t32[k][2] * EEO[2] +
			t32[k][3] * EEO[3] + add) >> shift);
	}
	else if (k % 4 == 2)//2 6 10 14 18 22 26 30
	{
		dst[k * line] = (int16_t)((t32[k][0] * EO[0] + t32[k][1] * EO[1] + t32[k][2] * EO[2] +
			t32[k][3] * EO[3] + t32[k][4] * EO[4] + t32[k][5] * EO[5] +
			t32[k][6] * EO[6] + t32[k][7] * EO[7] + add) >> shift);
	}
	else if (k % 2 == 1)//1 3 5 7 9 - odds 
	{
		dst[k * line] = (int16_t)((t32[k][0] * O[0] + t32[k][1] * O[1] + t32[k][2] * O[2] + t32[k][3] * O[3] +
			t32[k][4] * O[4] + t32[k][5] * O[5] + t32[k][6] * O[6] + t32[k][7] * O[7] +
			t32[k][8] * O[8] + t32[k][9] * O[9] + t32[k][10] * O[10] + t32[k][11] *
			O[11] + t32[k][12] * O[12] + t32[k][13] * O[13] + t32[k][14] * O[14] +
			t32[k][15] * O[15] + add) >> shift);
	}
}

__global__ void partialButterfly8(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j = blockIdx.x;// max 8
	int k = threadIdx.x;//max 8
	__shared__ int E[4], O[4];
	int EE[2], EO[2];
	int add = 1 << (shift - 1);

	src += 8 * j;
	dst += j;

	/* E and O*/
	if (k < 4)
	{
		E[k] = src[k] + src[7 - k];
		O[k] = src[k] - src[7 - k];
	}

	__syncthreads();

	/* EE and EO */
	EE[0] = E[0] + E[3];
	EO[0] = E[0] - E[3];
	EE[1] = E[1] + E[2];
	EO[1] = E[1] - E[2];

	dst[0] = (int16_t)((t8[0][0] * EE[0] + t8[0][1] * EE[1] + add) >> shift);
	dst[4 * line] = (int16_t)((t8[4][0] * EE[0] + t8[4][1] * EE[1] + add) >> shift);
	dst[2 * line] = (int16_t)((t8[2][0] * EO[0] + t8[2][1] * EO[1] + add) >> shift);
	dst[6 * line] = (int16_t)((t8[6][0] * EO[0] + t8[6][1] * EO[1] + add) >> shift);

	dst[line] = (int16_t)((t8[1][0] * O[0] + t8[1][1] * O[1] + t8[1][2] * O[2] + t8[1][3] * O[3] + add) >> shift);
	dst[3 * line] = (int16_t)((t8[3][0] * O[0] + t8[3][1] * O[1] + t8[3][2] * O[2] + t8[3][3] * O[3] + add) >> shift);
	dst[5 * line] = (int16_t)((t8[5][0] * O[0] + t8[5][1] * O[1] + t8[5][2] * O[2] + t8[5][3] * O[3] + add) >> shift);
	dst[7 * line] = (int16_t)((t8[7][0] * O[0] + t8[7][1] * O[1] + t8[7][2] * O[2] + t8[7][3] * O[3] + add) >> shift);
}

__global__ void partialButterflyInverse4(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j = threadIdx.x;
	int E[2], O[2];
	int add = 1 << (shift - 1);

	dst += 4 * j;
	src += j;

	/* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
	O[0] = t4[1][0] * src[line] + t4[3][0] * src[3 * line];
	O[1] = t4[1][1] * src[line] + t4[3][1] * src[3 * line];
	E[0] = t4[0][0] * src[0] + t4[2][0] * src[2 * line];
	E[1] = t4[0][1] * src[0] + t4[2][1] * src[2 * line];

	/* Combining even and odd terms at each hierarchy levels to calculate the final spatial domain vector */
	dst[0] = (int16_t)(g_x265_clip3(-32768, 32767, (E[0] + O[0] + add) >> shift));
	dst[1] = (int16_t)(g_x265_clip3(-32768, 32767, (E[1] + O[1] + add) >> shift));
	dst[2] = (int16_t)(g_x265_clip3(-32768, 32767, (E[1] - O[1] + add) >> shift));
	dst[3] = (int16_t)(g_x265_clip3(-32768, 32767, (E[0] - O[0] + add) >> shift));
}

__global__ void partialButterflyInverse8(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j = blockIdx.x;// max 8
	int k = threadIdx.x;//max 8
	__shared__ int E[4], O[4];
	int EE[2], EO[2];
	int add = 1 << (shift - 1);

	dst += 8 * j;
	src += j;

	/* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
	if (k < 4)
	{
		O[k] = t8[1][k] * src[line] + t8[3][k] * src[3 * line] + t8[5][k] * src[5 * line] + t8[7][k] * src[7 * line];
	}

	if (k == 5)//unused thread
	{
		EO[0] = t8[2][0] * src[2 * line] + t8[6][0] * src[6 * line];
		EO[1] = t8[2][1] * src[2 * line] + t8[6][1] * src[6 * line];
		EE[0] = t8[0][0] * src[0] + t8[4][0] * src[4 * line];
		EE[1] = t8[0][1] * src[0] + t8[4][1] * src[4 * line];

		/* Combining even and odd terms at each hierarchy levels to calculate the final spatial domain vector */
		E[0] = EE[0] + EO[0];
		E[3] = EE[0] - EO[0];
		E[1] = EE[1] + EO[1];
		E[2] = EE[1] - EO[1];
	}
	__syncthreads();
	if (k < 4)
	{
		dst[k] = (int16_t)g_x265_clip3(-32768, 32767, (E[k] + O[k] + add) >> shift);
		dst[k + 4] = (int16_t)g_x265_clip3(-32768, 32767, (E[3 - k] - O[3 - k] + add) >> shift);
	}
}

__global__ void partialButterflyInverse16(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j = blockIdx.x;// max 16
	int k = threadIdx.x;//max 16
	__shared__ int E[8], O[8];
	__shared__ int EE[4], EO[4];
	__shared__ int EEE[2], EEO[2];
	int add = 1 << (shift - 1);

	dst += 16 * j;
	src += j;

	/* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
	if (k < 8)
	{
		O[k] = t16[1][k] * src[line] + t16[3][k] * src[3 * line] + t16[5][k] * src[5 * line] + t16[7][k] * src[7 * line] +
			t16[9][k] * src[9 * line] + t16[11][k] * src[11 * line] + t16[13][k] * src[13 * line] + t16[15][k] * src[15 * line];
	}
	else if (k < 12)
	{
		int i = k - 8;
		EO[i] = t16[2][i] * src[2 * line] + t16[6][i] * src[6 * line] + t16[10][i]
			* src[10 * line] + t16[14][i] * src[14 * line];
	}
	else if (k == 12){
		EEO[0] = t16[4][0] * src[4 * line] + t16[12][0] * src[12 * line];
		EEE[0] = t16[0][0] * src[0] + t16[8][0] * src[8 * line];
		EEO[1] = t16[4][1] * src[4 * line] + t16[12][1] * src[12 * line];
		EEE[1] = t16[0][1] * src[0] + t16[8][1] * src[8 * line];
	}

	__syncthreads();
	/* Combining even and odd terms at each hierarchy levels to calculate the final spatial domain vector */
	if (k < 2)
	{
		EE[k] = EEE[k] + EEO[k];
		EE[k + 2] = EEE[1 - k] - EEO[1 - k];
	}
	__syncthreads();
	if (k < 4)
	{
		E[k] = EE[k] + EO[k];
		E[k + 4] = EE[3 - k] - EO[3 - k];
	}
	__syncthreads();
	if (k < 8)
	{
		dst[k] = (int16_t)g_x265_clip3(-32768, 32767, (E[k] + O[k] + add) >> shift);
		dst[k + 8] = (int16_t)g_x265_clip3(-32768, 32767, (E[7 - k] - O[7 - k] + add) >> shift);
	}
}

__global__ void partialButterflyInverse32(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j = blockIdx.x;// max 32
	int k = threadIdx.x;//max 32
	__shared__ int E[16], O[16];
	__shared__ int EE[8], EO[8];
	__shared__ int EEE[4], EEO[4];
	__shared__ int EEEE[2], EEEO[2];
	int add = 1 << (shift - 1);

	dst += 32 * j;
	src += j;

	/* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
	if (k < 16)
	{
		O[k] = t32[1][k] * src[line] + t32[3][k] * src[3 * line] + t32[5][k] * src[5 * line] + t32[7][k] * src[7 * line] +
			t32[9][k] * src[9 * line] + t32[11][k] * src[11 * line] + t32[13][k] * src[13 * line] + t32[15][k] * src[15 * line] +
			t32[17][k] * src[17 * line] + t32[19][k] * src[19 * line] + t32[21][k] * src[21 * line] + t32[23][k] * src[23 * line] +
			t32[25][k] * src[25 * line] + t32[27][k] * src[27 * line] + t32[29][k] * src[29 * line] + t32[31][k] * src[31 * line];
	}
	else if (k < 24)
	{
		int i = k - 16;
		EO[i] = t32[2][i] * src[2 * line] + t32[6][i] * src[6 * line] + t32[10][i] * src[10 * line] + t32[14][i] * src[14 * line] +
			t32[18][i] * src[18 * line] + t32[22][i] * src[22 * line] + t32[26][i] * src[26 * line] + t32[30][i] * src[30 * line];
	}

	else if (k < 28)
	{
		int i = k - 24;
		EEO[i] = t32[4][i] * src[4 * line] + t32[12][i] * src[12 * line] + t32[20][i] * src[20 * line] + t32[28][i] * src[28 * line];
	}
	else if (k == 28)
	{
		EEEO[0] = t32[8][0] * src[8 * line] + t32[24][0] * src[24 * line];
		EEEO[1] = t32[8][1] * src[8 * line] + t32[24][1] * src[24 * line];
		EEEE[0] = t32[0][0] * src[0] + t32[16][0] * src[16 * line];
		EEEE[1] = t32[0][1] * src[0] + t32[16][1] * src[16 * line];

		/* Combining even and odd terms at each hierarchy levels to calculate the final spatial domain vector */
		EEE[0] = EEEE[0] + EEEO[0];
		EEE[3] = EEEE[0] - EEEO[0];
		EEE[1] = EEEE[1] + EEEO[1];
		EEE[2] = EEEE[1] - EEEO[1];
	}
	__syncthreads();
	if (k < 4)
	{
		EE[k] = EEE[k] + EEO[k];
		EE[k + 4] = EEE[3 - k] - EEO[3 - k];
	}
	__syncthreads();
	if (k < 8)
	{
		E[k] = EE[k] + EO[k];
		E[k + 8] = EE[7 - k] - EO[7 - k];
	}
	__syncthreads();
	if (k < 16)
	{
		dst[k] = (int16_t)g_x265_clip3(-32768, 32767, (E[k] + O[k] + add) >> shift);
		dst[k + 16] = (int16_t)g_x265_clip3(-32768, 32767, (E[15 - k] - O[15 - k] + add) >> shift);
	}
}

__global__ void partialButterfly4(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j = threadIdx.x;
	int E[2], O[2];
	int add = 1 << (shift - 1);

	src += 4 * j;
	dst += j;

	/* E and O */
	E[0] = src[0] + src[3];
	O[0] = src[0] - src[3];
	E[1] = src[1] + src[2];
	O[1] = src[1] - src[2];

	dst[0] = (int16_t)((t4[0][0] * E[0] + t4[0][1] * E[1] + add) >> shift);
	dst[2 * line] = (int16_t)((t4[2][0] * E[0] + t4[2][1] * E[1] + add) >> shift);
	dst[line] = (int16_t)((t4[1][0] * O[0] + t4[1][1] * O[1] + add) >> shift);
	dst[3 * line] = (int16_t)((t4[3][0] * O[0] + t4[3][1] * O[1] + add) >> shift);
}

//template<FTYPESTAGE type>
//__global__ void transformAtomic(const int16_t *src, int16_t *dst, int n, const int shift) {
//	int col = blockIdx.x;
//	int row = threadIdx.x;
//	int mid = threadIdx.y;
//	extern __shared__ int16_t shared[];
//	shared[row * n + mid] = src[row * n + mid];
//	int size = n*n;
//	int size2 = size * 2;
//	int size3 = size * 3;
//	//sum
//	int *sum = (int*)&shared[size2 + row];
//	sum[row] = 0;
//	//counter
//	int *count = (int*)&shared[size3 + row];
//	count[row] = 0;
//	switch (type)
//	{
//	case DSTN:
//	case DSTT:
//	case IDSTT:
//	case IDSTN:
//		shared[size + row*n + mid] = ta[row][mid];
//		break;
//	case DCT4N:
//	case DCT4T:
//	case IDCT4N:
//	case IDCT4T:
//		shared[size + row*n + mid] = t4[row][mid];
//		break;
//	case DCT8N:
//	case DCT8T:
//	case IDCT8N:
//	case IDCT8T:
//		shared[size + row*n + mid] = t8[row][mid];
//		break;
//	case DCT16N:
//	case DCT16T:
//	case IDCT16N:
//	case IDCT16T:
//		shared[size + row*n + mid] = t16[row][mid];
//		break;
//	case DCT32N:
//	case DCT32T:
//	case IDCT32N:
//	case IDCT32T:
//		shared[size + row*n + mid] = t32[row][mid];
//		break;
//	}
//	__syncthreads();
//
//	int mul;
//	switch (type)
//	{
//	case DSTN:
//	case DCT4N:
//	case DCT8N:
//	case DCT16N:
//	case DCT32N:
//		mul = shared[size + row*n + mid] * shared[mid * n + col];
//		break;
//	case DSTT:
//	case DCT4T:
//	case DCT8T:
//	case DCT16T:
//	case DCT32T:
//		mul = shared[row * n + mid] * shared[size + col*n + mid];
//		break;
//	case IDSTN:
//	case IDCT4N:
//	case IDCT8N:
//	case IDCT16N:
//	case IDCT32N:
//		mul = shared[size + mid*n + row] * shared[mid * n + col];
//		break;
//	case IDSTT:
//	case IDCT4T:
//	case IDCT8T:
//	case IDCT16T:
//	case IDCT32T:
//		mul = shared[row * n + mid] * shared[size + mid*n + col];
//		break;
//	default:
//		mul = 0;
//	}
//	//sum
//	atomicAdd(&sum[row], mul);
//	//counter
//	atomicAdd(&count[row], 1);
//	if (count[row] >= n){
//		sum[row] >>= shift;
//		dst[row * n + col] = (int16_t)sum[row];
//	}
//}

template<FTYPE ftype>
__global__ void transform1stepBatch(const int16_t *src, int16_t *dst, const int n, const int shift1, const int shift2) {
	int row = threadIdx.y;
	int col = threadIdx.x;
	int m = blockIdx.x;
	int size = n*n;
	extern __shared__ int16_t shared[];
	shared[row * n + col] = src[m*size+row * n + col];
	int size2 = size * 2;
	switch (ftype)
	{
	case DST:
	case IDST:
		shared[size + row*n + col] = ta[row][col];
		break;
	case DCT4:
	case IDCT4:
		shared[size + row*n + col] = t4[row][col];
		break;
	case DCT8:
	case IDCT8:
		shared[size + row*n + col] = t8[row][col];
		break;
	case DCT16:
	case IDCT16:
		shared[size + row*n + col] = t16[row][col];
		break;
	case DCT32:
	case IDCT32:
		shared[size + row*n + col] = t32[row][col];
		break;
	}
	__syncthreads();
	int sum = 0;
	for (int i = 0; i < n; i++)
	{
		switch (ftype)
		{
		case DST:
		case DCT4:
		case DCT8:
		case DCT16:
		case DCT32:
			sum += shared[size + row*n + i] * shared[i * n + col];
			break;
		case IDST:
		case IDCT4:
		case IDCT8:
		case IDCT16:
		case IDCT32:
			sum += shared[size + i*n + row] * shared[i * n + col];
			break;
		}
	}
	sum >>= shift1;
	shared[size2 + row * n + col] = sum;
	__syncthreads();

	for (int i = 0; i < n; i++)
	{
		switch (ftype)
		{
		case DST:
		case DCT4:
		case DCT8:
		case DCT16:
		case DCT32:
			sum += shared[size2 + row * n + i] * shared[size + col*n + i];
			break;
		case IDST:
		case IDCT4:
		case IDCT8:
		case IDCT16:
		case IDCT32:
			sum += shared[size2 + row * n + i] * shared[size + i*n + col];
			break;
		}
	}
	sum >>= shift2;
	dst[m*size + row * n + col] = sum;
}

template<FTYPE ftype>
__global__ void transform1step(const int16_t *src, int16_t *dst, const int n, const int shift1, const int shift2) {
	int row = threadIdx.y;
	int col = threadIdx.x;
	extern __shared__ int16_t shared[];
	shared[row * n + col] = src[row * n + col];
	int size = n*n;
	int size2 = size * 2;
	switch (ftype)
	{
	case DST:
	case IDST:
		shared[size + row*n + col] = ta[row][col];
		break;
	case DCT4:
	case IDCT4:
		shared[size + row*n + col] = t4[row][col];
		break;
	case DCT8:
	case IDCT8:
		shared[size + row*n + col] = t8[row][col];
		break;
	case DCT16:
	case IDCT16:
		shared[size + row*n + col] = t16[row][col];
		break;
	case DCT32:
	case IDCT32:
		shared[size + row*n + col] = t32[row][col];
		break;
	}
	__syncthreads();
	int sum = 0;
	for (int i = 0; i < n; i++)
	{
		switch (ftype)
		{
		case DST:
		case DCT4:
		case DCT8:
		case DCT16:
		case DCT32:
			sum += shared[size + row*n + i] * shared[i * n + col];
			break;
		case IDST:
		case IDCT4:
		case IDCT8:
		case IDCT16:
		case IDCT32:
			sum += shared[size + i*n + row] * shared[i * n + col];
			break;
		}
	}
	sum >>= shift1;
	shared[size2 + row * n + col] = sum;
	__syncthreads();

	for (int i = 0; i < n; i++)
	{
		switch (ftype)
		{
		case DST:
		case DCT4:
		case DCT8:
		case DCT16:
		case DCT32:
			sum += shared[size2 + row * n + i] * shared[size + col*n + i];
			break;
		case IDST:
		case IDCT4:
		case IDCT8:
		case IDCT16:
		case IDCT32:
			sum += shared[size2 + row * n + i] * shared[size + i*n + col];
			break;
		}
	}
	sum >>= shift2;
	dst[row * n + col] = sum;
}

__global__ void dct32_g(const int16_t *src, int16_t *dst ,const int shift1, const int shift2) {
	int row = threadIdx.y;
	int col = threadIdx.x;
	const int n = 32;
	extern __shared__ int16_t shared[];
	shared[row * n + col] = src[row * n + col];
	int size = n*n;
	int size2 = size * 2;
	shared[size + row*n + col] = t32[row][col];
	__syncthreads();
	int sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += shared[size + row*n + i] * shared[i * n + col];
	}
	sum >>= shift1;
	shared[size2 + row * n + col] = sum;
	__syncthreads();

	for (int i = 0; i < n; i++)
	{
		sum += shared[size2 + row * n + i] * shared[size + col*n + i];
	}
	sum >>= shift2;
	dst[row * n + col] = sum;
}

__global__ void idct32_g(const int16_t *src, int16_t *dst, const int shift1, const int shift2) {
	int row = threadIdx.y;
	int col = threadIdx.x;
	const int n = 32;
	extern __shared__ int16_t shared[];
	shared[row * n + col] = src[row * n + col];
	int size = n*n;
	int size2 = size * 2;
	shared[size + row*n + col] = t32[row][col];
	__syncthreads();
	int sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += shared[size + i*n + row] * shared[i * n + col];
	}
	sum >>= shift1;
	shared[size2 + row * n + col] = sum;
	__syncthreads();

	for (int i = 0; i < n; i++)
	{
		sum += shared[size2 + row * n + i] * shared[size + i*n + col];
	}
	sum >>= shift2;
	dst[row * n + col] = sum;
}

template<FTYPESTAGE type>
__global__ void transform(const int16_t *src, int16_t *dst, int n, const int shift) {
	int row = threadIdx.y;
	int col = threadIdx.x;
	extern __shared__ int16_t shared[];
	shared[row * n + col] = src[row * n + col];
	int size = n*n;
	switch (type)
	{
	case DSTN:
	case DSTT:
	case IDSTT:
	case IDSTN:
		shared[size + row*n + col] = ta[row][col];
		break;
	case DCT4N:
	case DCT4T:
	case IDCT4N:
	case IDCT4T:
		shared[size + row*n + col] = t4[row][col];
		break;
	case DCT8N:
	case DCT8T:
	case IDCT8N:
	case IDCT8T:
		shared[size + row*n + col] = t8[row][col];
		break;
	case DCT16N:
	case DCT16T:
	case IDCT16N:
	case IDCT16T:
		shared[size + row*n + col] = t16[row][col];
		break;
	case DCT32N:
	case DCT32T:
	case IDCT32N:
	case IDCT32T:
		shared[size + row*n + col] = t32[row][col];
		break;
	}
	__syncthreads();
	int sum = 0;
	for (int i = 0; i < n; i++)
	{
		switch (type)
		{
		case DSTN:
		case DCT4N:
		case DCT8N:
		case DCT16N:
		case DCT32N:
			sum += shared[size + row*n + i] * shared[i * n + col];
			break;
		case DSTT:
		case DCT4T:
		case DCT8T:
		case DCT16T:
		case DCT32T:
			sum += shared[row * n + i] * shared[size + col*n + i];
			break;
		case IDSTN:
		case IDCT4N:
		case IDCT8N:
		case IDCT16N:
		case IDCT32N:
			sum += shared[size + i*n + row] * shared[i * n + col];
			break;
		case IDSTT:
		case IDCT4T:
		case IDCT8T:
		case IDCT16T:
		case IDCT32T:
			sum += shared[row * n + i] * shared[size + i*n + col];
			break;
		}
	}
	sum >>= shift;
	dst[row * n + col] = sum;
}

template<FTYPESTAGE type>
__global__ void transformPlain(const int16_t *src, int16_t *dst, int n, const int shift) {
	int row = threadIdx.y;
	int col = threadIdx.x;
	int sum = 0;
	for (int i = 0; i < n; i++)
	{
		switch (type)
		{
		case DSTN:
			sum += ta[row][i] * src[i * n + col];
			break;
		case DSTT:
			sum += src[row * n + i] * ta[col][i];
			break;
		case DCT4N:
			sum += t4[row][i] * src[i * n + col];
			break;
		case DCT4T:
			sum += src[row * n + i] * t4[col][i];
			break;
		case DCT8N:
			sum += t8[row][i] * src[i * n + col];
			break;
		case DCT8T:
			sum += src[row * n + i] * t8[col][i];
			break;
		case DCT16N:
			sum += t16[row][i] * src[i * n + col];
			break;
		case DCT16T:
			sum += src[row * n + i] * t16[col][i];
			break;
		case DCT32N:
			sum += t32[row][i] * src[i * n + col];
			break;
		case DCT32T:
			sum += src[row * n + i] * t32[col][i];
			break;
		case IDSTN:
			sum += ta[i][row] * src[i * n + col];
			break;
		case IDSTT:
			sum += src[row * n + i] * ta[i][col];
			break;
		case IDCT4N:
			sum += t4[i][row] * src[i * n + col];
			break;
		case IDCT4T:
			sum += src[row * n + i] * t4[i][col];
			break;
		case IDCT8N:
			sum += t8[i][row] * src[i * n + col];
			break;
		case IDCT8T:
			sum += src[row * n + i] * t8[i][col];
			break;
		case IDCT16N:
			sum += t16[i][row] * src[i * n + col];
			break;
		case IDCT16T:
			sum += src[row * n + i] * t16[i][col];
			break;
		case IDCT32N:
			sum += t32[i][row] * src[i * n + col];
			break;
		case IDCT32T:
			sum += src[row * n + i] * t32[i][col];
			break;
		}
	}
	sum >>= shift;
	dst[row * n + col] = sum;
}

template __global__ void transformPlain<DSTN>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<DSTT>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<DCT4N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<DCT4T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<DCT8N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<DCT8T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<DCT16N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<DCT16T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<DCT32N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<DCT32T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<IDSTN>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<IDSTT>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<IDCT4N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<IDCT4T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<IDCT8N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<IDCT8T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<IDCT16N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<IDCT16T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<IDCT32N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transformPlain<IDCT32T>(const int16_t *src, int16_t *dst, int n, const int shift);

template __global__ void transform<DSTN>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<DSTT>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<DCT4N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<DCT4T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<DCT8N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<DCT8T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<DCT16N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<DCT16T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<DCT32N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<DCT32T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<IDSTN>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<IDSTT>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<IDCT4N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<IDCT4T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<IDCT8N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<IDCT8T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<IDCT16N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<IDCT16T>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<IDCT32N>(const int16_t *src, int16_t *dst, int n, const int shift);
template __global__ void transform<IDCT32T>(const int16_t *src, int16_t *dst, int n, const int shift);

template __global__ void transform1step<DST>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1step<DCT4>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1step<DCT8>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1step<DCT16>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1step<DCT32>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1step<IDST>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1step<IDCT4>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1step<IDCT8>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1step<IDCT16>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1step<IDCT32>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);

template __global__ void transform1stepBatch<DST>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1stepBatch<DCT4>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1stepBatch<DCT8>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1stepBatch<DCT16>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1stepBatch<DCT32>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1stepBatch<IDST>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1stepBatch<IDCT4>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1stepBatch<IDCT8>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1stepBatch<IDCT16>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);
template __global__ void transform1stepBatch<IDCT32>(const int16_t *src, int16_t *dst, int n, const int shift1, const int shift2);

template<FTYPE type>
void gpuTransformPlain(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n) {
	// Allocate 3 arrays on GPU
	//int16_t *dst, *src, *src2;
	//cudaMalloc(&src, n * n * sizeof(int16_t));
	//cudaMalloc(&src2, n * n * sizeof(int16_t));
	//cudaMalloc(&dst, n * n * sizeof(int16_t));

	cudaMemcpy(src, h_src, n * n * sizeof(int16_t), cudaMemcpyHostToDevice);

	dim3 dimGrid(1, 1);
	dim3  dimBlock(n, n);
	switch (type)
	{
	case DST:
		transformPlain<DSTN> << <dimGrid, dimBlock >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transformPlain<DSTT> << <dimGrid, dimBlock >> >(src2, dst, n, shift2);
		break;
	case DCT4:
		transformPlain<DCT4N> << <dimGrid, dimBlock >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transformPlain<DCT4T> << <dimGrid, dimBlock >> >(src2, dst, n, shift2);
		break;
	case DCT8:
		transformPlain<DCT8N> << <dimGrid, dimBlock >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transformPlain<DCT8T> << <dimGrid, dimBlock >> >(src2, dst, n, shift2);
		break;
	case DCT16:
		transformPlain<DCT16N> << <dimGrid, dimBlock>> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transformPlain<DCT16T> << <dimGrid, dimBlock>> >(src2, dst, n, shift2);
		break;
	case DCT32:
		transformPlain<DCT32N> << <dimGrid, dimBlock >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transformPlain<DCT32T> << <dimGrid, dimBlock >> >(src2, dst, n, shift2);
		break;
	case IDST:
		transformPlain<IDSTN> << <dimGrid, dimBlock >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transformPlain<IDSTT> << <dimGrid, dimBlock >> >(src2, dst, n, shift2);
		break;
	case IDCT4:
		transformPlain<IDCT4N> << <dimGrid, dimBlock >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transformPlain<IDCT4T> << <dimGrid, dimBlock >> >(src2, dst, n, shift2);
		break;
	case IDCT8:
		transformPlain<IDCT8N> << <dimGrid, dimBlock >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transformPlain<IDCT8T> << <dimGrid, dimBlock >> >(src2, dst, n, shift2);
		break;
	case IDCT16:
		transformPlain<IDCT16N> << <dimGrid, dimBlock >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transformPlain<IDCT16T> << <dimGrid, dimBlock >> >(src2, dst, n, shift2);
		break;
	case IDCT32:
		transformPlain<IDCT32N> << <dimGrid, dimBlock >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transformPlain<IDCT32T> << <dimGrid, dimBlock >> >(src2, dst, n, shift2);
		break;
	}

	// Copy (and print) the result on host memory
	cudaMemcpy(h_dst, dst, n * n * sizeof(int16_t), cudaMemcpyDeviceToHost);

	//Free GPU memory
	//cudaFree(src);
	//cudaFree(src2);
	//cudaFree(dst);
}

template<FTYPE type>
void gpuTransformShared(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n) {
	// Allocate 3 arrays on GPU
	//int16_t *dst, *src, *src2;
	//cudaMalloc(&src, n * n * sizeof(int16_t));
	//cudaMalloc(&src2, n * n * sizeof(int16_t));
	//cudaMalloc(&dst, n * n * sizeof(int16_t));

	cudaMemcpy(src, h_src, n * n * sizeof(int16_t), cudaMemcpyHostToDevice);

	dim3 dimGrid(1);
	dim3  dimBlock(n, n);
	int size = 2 * n*n* sizeof(int16_t);
	switch (type)
	{
	case DST:
		transform<DSTN> << <dimGrid, dimBlock, size >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transform<DSTT> << <dimGrid, dimBlock, size >> >(src2, dst, n, shift2);
		break;
	case DCT4:
		transform<DCT4N> << <dimGrid, dimBlock, size >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transform<DCT4T> << <dimGrid, dimBlock, size >> >(src2, dst, n, shift2);
		break;
	case DCT8:
		transform<DCT8N> << <dimGrid, dimBlock, size >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transform<DCT8T> << <dimGrid, dimBlock, size >> >(src2, dst, n, shift2);
		break;
	case DCT16:
		transform<DCT16N> << <dimGrid, dimBlock, size >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transform<DCT16T> << <dimGrid, dimBlock, size >> >(src2, dst, n, shift2);
		break;
	case DCT32:
		transform<DCT32N> << <dimGrid, dimBlock, size >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transform<DCT32T> << <dimGrid, dimBlock, size >> >(src2, dst, n, shift2);
		break;
	case IDST:
		transform<IDSTN> << <dimGrid, dimBlock, size >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transform<IDSTT> << <dimGrid, dimBlock, size >> >(src2, dst, n, shift2);
		break;
	case IDCT4:
		transform<IDCT4N> << <dimGrid, dimBlock, size >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transform<IDCT4T> << <dimGrid, dimBlock, size >> >(src2, dst, n, shift2);
		break;
	case IDCT8:
		transform<IDCT8N> << <dimGrid, dimBlock, size >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transform<IDCT8T> << <dimGrid, dimBlock, size >> >(src2, dst, n, shift2);
		break;
	case IDCT16:
		transform<IDCT16N> << <dimGrid, dimBlock, size >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transform<IDCT16T> << <dimGrid, dimBlock, size >> >(src2, dst, n, shift2);
		break;
	case IDCT32:
		transform<IDCT32N> << <dimGrid, dimBlock, size >> >(src, src2, n, shift1);
		cudaDeviceSynchronize();
		transform<IDCT32T> << <dimGrid, dimBlock, size >> >(src2, dst, n, shift2);
		break;
	}

	// Copy (and print) the result on host memory
	cudaMemcpy(h_dst, dst, n * n * sizeof(int16_t), cudaMemcpyDeviceToHost);

	//Free GPU memory
	//cudaFree(src);
	//cudaFree(src2);
	//cudaFree(dst);
}

template<FTYPE type>
void gpuTransform1Step(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n) {
	// Allocate 3 arrays on GPU
	//int16_t *dst, *src;
	//cudaMalloc(&src, n * n * sizeof(int16_t));
	//cudaMalloc(&dst, n * n * sizeof(int16_t));

	cudaMemcpy(src, h_src, n * n * sizeof(int16_t), cudaMemcpyHostToDevice);

	dim3 dimGrid(1);
	dim3  dimBlock(n, n);
	int size = 3 * n*n* sizeof(int16_t);
	switch (type)
	{
	case DST:
		transform1step<DST> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case DCT4:
		transform1step<DCT4> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case DCT8:
		transform1step<DCT8> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case DCT16:
		transform1step<DCT16> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case DCT32:
		dct32_g << <dimGrid, dimBlock, size >> >(src, dst, shift1, shift2);
		break;
	case IDST:
		transform1step<IDST> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case IDCT4:
		transform1step<IDCT4> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case IDCT8:
		transform1step<IDCT8> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case IDCT16:
		transform1step<IDCT16> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case IDCT32:
		idct32_g<< <dimGrid, dimBlock, size >> >(src, dst, shift1, shift2);
		break;
	}

	// Copy (and print) the result on host memory
	cudaMemcpy(h_dst, dst, n * n * sizeof(int16_t), cudaMemcpyDeviceToHost);

	//Free GPU memory
	//cudaFree(src);
	//cudaFree(dst);
}

template<FTYPE ftype>
void gpuTransform1StepBatch(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n,int m) {
	// Allocate 3 arrays on GPU
	//int16_t *dst, *src, *src2;
	//cudaMalloc(&src, n * n * sizeof(int16_t));
	//cudaMalloc(&src2, n * n * sizeof(int16_t));
	//cudaMalloc(&dst, n * n * sizeof(int16_t));

	cudaMemcpy(src, h_src, m * n * n * sizeof(int16_t), cudaMemcpyHostToDevice);

	dim3 dimGrid(m);
	dim3  dimBlock(n, n);
	int size = 3 * n*n* sizeof(int16_t);
	switch (ftype)
	{
	case DST:
		transform1stepBatch<DST> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case DCT4:
		transform1stepBatch<DCT4> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case DCT8:
		transform1stepBatch<DCT8> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case DCT16:
		transform1stepBatch<DCT16> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case DCT32:
		transform1stepBatch<DCT32> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case IDST:
		transform1stepBatch<IDST> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case IDCT4:
		transform1stepBatch<IDCT4> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case IDCT8:
		transform1stepBatch<IDCT8> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case IDCT16:
		transform1stepBatch<IDCT16> << <dimGrid, dimBlock, size >> >(src, dst, n, shift1, shift2);
		break;
	case IDCT32:
		transform1stepBatch<IDCT32> << <dimGrid, dimBlock, size >> >(src, dst,n, shift1, shift2);
		break;
	}

	// Copy (and print) the result on host memory
	cudaMemcpy(h_dst, dst, m * n * n * sizeof(int16_t), cudaMemcpyDeviceToHost);

	//Free GPU memory
	//cudaFree(src);
	//cudaFree(src2);
	//cudaFree(dst);
}

template<FTYPE type>
void gpuLessMulTransform(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n) {
	// Allocate 3 arrays on GPU
	//int16_t *dst, *src, *src2;
	//cudaMalloc(&src, n * n * sizeof(int16_t));
	//cudaMalloc(&src2, n * n * sizeof(int16_t));
	//cudaMalloc(&dst, n * n * sizeof(int16_t));

	cudaMemcpy(src, h_src, n * n * sizeof(int16_t), cudaMemcpyHostToDevice);

	dim3 dimGrid(n);
	dim3  dimBlock(n);

	switch (type)
	{
	case DST:
		fastForwardDst << <1, n>> >(src, src2,shift1);
		cudaDeviceSynchronize();
		fastForwardDst << <1, n >> >(src2, dst, shift2);
		break;
	case DCT4:
		partialButterfly4 << <1, n >> >(src, src2, shift1, n);
		cudaDeviceSynchronize();
		partialButterfly4 << <1, n >> >(src2, dst, shift2, n);
		break;
	case DCT8:
		partialButterfly8 << <dimGrid, dimBlock >> >(src, src2, shift1, n);
		cudaDeviceSynchronize();
		partialButterfly8 << <dimGrid, dimBlock >> >(src2, dst, shift2, n);
		break;
	case DCT16:
		partialButterfly16 << <dimGrid, dimBlock >> >(src, src2, shift1, n);
		cudaDeviceSynchronize();
		partialButterfly16 << <dimGrid, dimBlock >> >(src2, dst, shift2, n);
		break;
	case DCT32:
		partialButterfly32 << <dimGrid, dimBlock >> >(src, src2, shift1, n);
		cudaDeviceSynchronize();
		partialButterfly32 << <dimGrid, dimBlock >> >(src2, dst, shift2, n);
		break;
	case IDST:
		inversedst << <1, n >> >(src, src2, shift1);
		cudaDeviceSynchronize();
		inversedst << <1, n >> >(src2, dst, shift2);
		break;
	case IDCT4:
		partialButterflyInverse4 << <1, n >> >(src, src2, shift1, n);
		cudaDeviceSynchronize();
		partialButterflyInverse4 << <1, n >> >(src2, dst, shift2, n);
		break;
	case IDCT8:
		partialButterflyInverse8 << <dimGrid, dimBlock >> >(src, src2, shift1, n);
		cudaDeviceSynchronize();
		partialButterflyInverse8 << <dimGrid, dimBlock >> >(src2, dst, shift2, n);
		break;
	case IDCT16:
		partialButterflyInverse16 << <dimGrid, dimBlock >> >(src, src2, shift1, n);
		cudaDeviceSynchronize();
		partialButterflyInverse16 << <dimGrid, dimBlock >> >(src2, dst, shift2, n);
		break;
	case IDCT32:
		partialButterflyInverse32 << <dimGrid, dimBlock >> >(src, src2, shift1, n);
		cudaDeviceSynchronize();
		partialButterflyInverse32 << <dimGrid, dimBlock >> >(src2, dst, shift2, n);
		break;
	}

	// Copy (and print) the result on host memory
	cudaMemcpy(h_dst, dst, n * n * sizeof(int16_t), cudaMemcpyDeviceToHost);

	//Free GPU memory
	//cudaFree(src);
	//cudaFree(src2);
	//cudaFree(dst);
}

void gpuTransform(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, FTYPE type) {
	int cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}
}