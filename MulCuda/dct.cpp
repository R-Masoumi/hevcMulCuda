#include "dct.h"

int16_t* generateArray(int n)
{
	short min = -128;
	short max = 128;
	int16_t* src = new int16_t[n * n];
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			src[i * n + j] = 2 * (min + (rand() % static_cast<short>(max - min + 1)));
		}
	}
	return src;
}

int16_t* generateArrayBatch(int n,int m)
{
	short min = -128;
	short max = 128;
	int16_t* src = new int16_t[n * n * m];
	for (int k = 0; k < m; ++k){
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				src[n*n*k+i * n + j] = 2 * (min + (rand() % static_cast<short>(max - min + 1)));
			}
		}
	}
	return src;
}

static void fastForwardDst(const int16_t* block, int16_t* coeff, int shift)  // input block, output coeff
{
	int c[4];
	int rnd_factor = 1 << (shift - 1);

	for (int i = 0; i < 4; i++)
	{
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
}

static void inversedst(const int16_t* tmp, int16_t* block, int shift)  // input tmp, output block
{
	int i, c[4];
	int rnd_factor = 1 << (shift - 1);

	for (i = 0; i < 4; i++)
	{
		// Intermediate Variables
		c[0] = tmp[i] + tmp[8 + i];
		c[1] = tmp[8 + i] + tmp[12 + i];
		c[2] = tmp[i] - tmp[12 + i];
		c[3] = 74 * tmp[4 + i];

		block[4 * i + 0] = (int16_t)x265_clip3(-32768, 32767, (29 * c[0] + 55 * c[1] + c[3] + rnd_factor) >> shift);
		block[4 * i + 1] = (int16_t)x265_clip3(-32768, 32767, (55 * c[2] - 29 * c[1] + c[3] + rnd_factor) >> shift);
		block[4 * i + 2] = (int16_t)x265_clip3(-32768, 32767, (74 * (tmp[i] - tmp[8 + i] + tmp[12 + i]) + rnd_factor) >> shift);
		block[4 * i + 3] = (int16_t)x265_clip3(-32768, 32767, (55 * c[0] + 29 * c[2] - c[3] + rnd_factor) >> shift);
	}
}

static void partialButterfly16(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j, k;
	int E[8], O[8];
	int EE[4], EO[4];
	int EEE[2], EEO[2];
	int add = 1 << (shift - 1);

	for (j = 0; j < line; j++)
	{
		/* E and O */
		for (k = 0; k < 8; k++)
		{
			E[k] = src[k] + src[15 - k];
			O[k] = src[k] - src[15 - k];
		}

		/* EE and EO */
		for (k = 0; k < 4; k++)
		{
			EE[k] = E[k] + E[7 - k];
			EO[k] = E[k] - E[7 - k];
		}

		/* EEE and EEO */
		EEE[0] = EE[0] + EE[3];
		EEO[0] = EE[0] - EE[3];
		EEE[1] = EE[1] + EE[2];
		EEO[1] = EE[1] - EE[2];

		dst[0] = (int16_t)((g_t16[0][0] * EEE[0] + g_t16[0][1] * EEE[1] + add) >> shift);
		dst[8 * line] = (int16_t)((g_t16[8][0] * EEE[0] + g_t16[8][1] * EEE[1] + add) >> shift);
		dst[4 * line] = (int16_t)((g_t16[4][0] * EEO[0] + g_t16[4][1] * EEO[1] + add) >> shift);
		dst[12 * line] = (int16_t)((g_t16[12][0] * EEO[0] + g_t16[12][1] * EEO[1] + add) >> shift);

		for (k = 2; k < 16; k += 4)
		{
			dst[k * line] = (int16_t)((g_t16[k][0] * EO[0] + g_t16[k][1] * EO[1] + g_t16[k][2] * EO[2] +
				g_t16[k][3] * EO[3] + add) >> shift);
		}

		for (k = 1; k < 16; k += 2)
		{
			dst[k * line] = (int16_t)((g_t16[k][0] * O[0] + g_t16[k][1] * O[1] + g_t16[k][2] * O[2] + g_t16[k][3] * O[3] +
				g_t16[k][4] * O[4] + g_t16[k][5] * O[5] + g_t16[k][6] * O[6] + g_t16[k][7] * O[7] +
				add) >> shift);
		}

		src += 16;
		dst++;
	}
}

static void partialButterfly32(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j, k;
	int E[16], O[16];
	int EE[8], EO[8];
	int EEE[4], EEO[4];
	int EEEE[2], EEEO[2];
	int add = 1 << (shift - 1);

	for (j = 0; j < line; j++)
	{
		/* E and O*/
		for (k = 0; k < 16; k++)
		{
			E[k] = src[k] + src[31 - k];
			O[k] = src[k] - src[31 - k];
		}

		/* EE and EO */
		for (k = 0; k < 8; k++)
		{
			EE[k] = E[k] + E[15 - k];
			EO[k] = E[k] - E[15 - k];
		}

		/* EEE and EEO */
		for (k = 0; k < 4; k++)
		{
			EEE[k] = EE[k] + EE[7 - k];
			EEO[k] = EE[k] - EE[7 - k];
		}

		/* EEEE and EEEO */
		EEEE[0] = EEE[0] + EEE[3];
		EEEO[0] = EEE[0] - EEE[3];
		EEEE[1] = EEE[1] + EEE[2];
		EEEO[1] = EEE[1] - EEE[2];

		dst[0] = (int16_t)((g_t32[0][0] * EEEE[0] + g_t32[0][1] * EEEE[1] + add) >> shift);
		dst[16 * line] = (int16_t)((g_t32[16][0] * EEEE[0] + g_t32[16][1] * EEEE[1] + add) >> shift);
		dst[8 * line] = (int16_t)((g_t32[8][0] * EEEO[0] + g_t32[8][1] * EEEO[1] + add) >> shift);
		dst[24 * line] = (int16_t)((g_t32[24][0] * EEEO[0] + g_t32[24][1] * EEEO[1] + add) >> shift);
		for (k = 4; k < 32; k += 8)
		{
			dst[k * line] = (int16_t)((g_t32[k][0] * EEO[0] + g_t32[k][1] * EEO[1] + g_t32[k][2] * EEO[2] +
				g_t32[k][3] * EEO[3] + add) >> shift);
		}

		for (k = 2; k < 32; k += 4)
		{
			dst[k * line] = (int16_t)((g_t32[k][0] * EO[0] + g_t32[k][1] * EO[1] + g_t32[k][2] * EO[2] +
				g_t32[k][3] * EO[3] + g_t32[k][4] * EO[4] + g_t32[k][5] * EO[5] +
				g_t32[k][6] * EO[6] + g_t32[k][7] * EO[7] + add) >> shift);
		}

		for (k = 1; k < 32; k += 2)
		{
			dst[k * line] = (int16_t)((g_t32[k][0] * O[0] + g_t32[k][1] * O[1] + g_t32[k][2] * O[2] + g_t32[k][3] * O[3] +
				g_t32[k][4] * O[4] + g_t32[k][5] * O[5] + g_t32[k][6] * O[6] + g_t32[k][7] * O[7] +
				g_t32[k][8] * O[8] + g_t32[k][9] * O[9] + g_t32[k][10] * O[10] + g_t32[k][11] *
				O[11] + g_t32[k][12] * O[12] + g_t32[k][13] * O[13] + g_t32[k][14] * O[14] +
				g_t32[k][15] * O[15] + add) >> shift);
		}

		src += 32;
		dst++;
	}
}

static void partialButterfly8(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j, k;
	int E[4], O[4];
	int EE[2], EO[2];
	int add = 1 << (shift - 1);

	for (j = 0; j < line; j++)
	{
		/* E and O*/
		for (k = 0; k < 4; k++)
		{
			E[k] = src[k] + src[7 - k];
			O[k] = src[k] - src[7 - k];
		}

		/* EE and EO */
		EE[0] = E[0] + E[3];
		EO[0] = E[0] - E[3];
		EE[1] = E[1] + E[2];
		EO[1] = E[1] - E[2];

		dst[0] = (int16_t)((g_t8[0][0] * EE[0] + g_t8[0][1] * EE[1] + add) >> shift);
		dst[4 * line] = (int16_t)((g_t8[4][0] * EE[0] + g_t8[4][1] * EE[1] + add) >> shift);
		dst[2 * line] = (int16_t)((g_t8[2][0] * EO[0] + g_t8[2][1] * EO[1] + add) >> shift);
		dst[6 * line] = (int16_t)((g_t8[6][0] * EO[0] + g_t8[6][1] * EO[1] + add) >> shift);

		dst[line] = (int16_t)((g_t8[1][0] * O[0] + g_t8[1][1] * O[1] + g_t8[1][2] * O[2] + g_t8[1][3] * O[3] + add) >> shift);
		dst[3 * line] = (int16_t)((g_t8[3][0] * O[0] + g_t8[3][1] * O[1] + g_t8[3][2] * O[2] + g_t8[3][3] * O[3] + add) >> shift);
		dst[5 * line] = (int16_t)((g_t8[5][0] * O[0] + g_t8[5][1] * O[1] + g_t8[5][2] * O[2] + g_t8[5][3] * O[3] + add) >> shift);
		dst[7 * line] = (int16_t)((g_t8[7][0] * O[0] + g_t8[7][1] * O[1] + g_t8[7][2] * O[2] + g_t8[7][3] * O[3] + add) >> shift);

		src += 8;
		dst++;
	}
}

static void partialButterflyInverse4(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j;
	int E[2], O[2];
	int add = 1 << (shift - 1);

	for (j = 0; j < line; j++)
	{
		/* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
		O[0] = g_t4[1][0] * src[line] + g_t4[3][0] * src[3 * line];
		O[1] = g_t4[1][1] * src[line] + g_t4[3][1] * src[3 * line];
		E[0] = g_t4[0][0] * src[0] + g_t4[2][0] * src[2 * line];
		E[1] = g_t4[0][1] * src[0] + g_t4[2][1] * src[2 * line];

		/* Combining even and odd terms at each hierarchy levels to calculate the final spatial domain vector */
		dst[0] = (int16_t)(x265_clip3(-32768, 32767, (E[0] + O[0] + add) >> shift));
		dst[1] = (int16_t)(x265_clip3(-32768, 32767, (E[1] + O[1] + add) >> shift));
		dst[2] = (int16_t)(x265_clip3(-32768, 32767, (E[1] - O[1] + add) >> shift));
		dst[3] = (int16_t)(x265_clip3(-32768, 32767, (E[0] - O[0] + add) >> shift));

		src++;
		dst += 4;
	}
}

static void partialButterflyInverse8(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j, k;
	int E[4], O[4];
	int EE[2], EO[2];
	int add = 1 << (shift - 1);

	for (j = 0; j < line; j++)
	{
		/* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
		for (k = 0; k < 4; k++)
		{
			O[k] = g_t8[1][k] * src[line] + g_t8[3][k] * src[3 * line] + g_t8[5][k] * src[5 * line] + g_t8[7][k] * src[7 * line];
		}

		EO[0] = g_t8[2][0] * src[2 * line] + g_t8[6][0] * src[6 * line];
		EO[1] = g_t8[2][1] * src[2 * line] + g_t8[6][1] * src[6 * line];
		EE[0] = g_t8[0][0] * src[0] + g_t8[4][0] * src[4 * line];
		EE[1] = g_t8[0][1] * src[0] + g_t8[4][1] * src[4 * line];

		/* Combining even and odd terms at each hierarchy levels to calculate the final spatial domain vector */
		E[0] = EE[0] + EO[0];
		E[3] = EE[0] - EO[0];
		E[1] = EE[1] + EO[1];
		E[2] = EE[1] - EO[1];
		for (k = 0; k < 4; k++)
		{
			dst[k] = (int16_t)x265_clip3(-32768, 32767, (E[k] + O[k] + add) >> shift);
			dst[k + 4] = (int16_t)x265_clip3(-32768, 32767, (E[3 - k] - O[3 - k] + add) >> shift);
		}

		src++;
		dst += 8;
	}
}

static void partialButterflyInverse16(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j, k;
	int E[8], O[8];
	int EE[4], EO[4];
	int EEE[2], EEO[2];
	int add = 1 << (shift - 1);

	for (j = 0; j < line; j++)
	{
		/* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
		for (k = 0; k < 8; k++)
		{
			O[k] = g_t16[1][k] * src[line] + g_t16[3][k] * src[3 * line] + g_t16[5][k] * src[5 * line] + g_t16[7][k] * src[7 * line] +
				g_t16[9][k] * src[9 * line] + g_t16[11][k] * src[11 * line] + g_t16[13][k] * src[13 * line] + g_t16[15][k] * src[15 * line];
		}

		for (k = 0; k < 4; k++)
		{
			EO[k] = g_t16[2][k] * src[2 * line] + g_t16[6][k] * src[6 * line] + g_t16[10][k] * src[10 * line] + g_t16[14][k] * src[14 * line];
		}

		EEO[0] = g_t16[4][0] * src[4 * line] + g_t16[12][0] * src[12 * line];
		EEE[0] = g_t16[0][0] * src[0] + g_t16[8][0] * src[8 * line];
		EEO[1] = g_t16[4][1] * src[4 * line] + g_t16[12][1] * src[12 * line];
		EEE[1] = g_t16[0][1] * src[0] + g_t16[8][1] * src[8 * line];

		/* Combining even and odd terms at each hierarchy levels to calculate the final spatial domain vector */
		for (k = 0; k < 2; k++)
		{
			EE[k] = EEE[k] + EEO[k];
			EE[k + 2] = EEE[1 - k] - EEO[1 - k];
		}

		for (k = 0; k < 4; k++)
		{
			E[k] = EE[k] + EO[k];
			E[k + 4] = EE[3 - k] - EO[3 - k];
		}

		for (k = 0; k < 8; k++)
		{
			dst[k] = (int16_t)x265_clip3(-32768, 32767, (E[k] + O[k] + add) >> shift);
			dst[k + 8] = (int16_t)x265_clip3(-32768, 32767, (E[7 - k] - O[7 - k] + add) >> shift);
		}

		src++;
		dst += 16;
	}
}

static void partialButterflyInverse32(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j, k;
	int E[16], O[16];
	int EE[8], EO[8];
	int EEE[4], EEO[4];
	int EEEE[2], EEEO[2];
	int add = 1 << (shift - 1);

	for (j = 0; j < line; j++)
	{
		/* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
		for (k = 0; k < 16; k++)
		{
			O[k] = g_t32[1][k] * src[line] + g_t32[3][k] * src[3 * line] + g_t32[5][k] * src[5 * line] + g_t32[7][k] * src[7 * line] +
				g_t32[9][k] * src[9 * line] + g_t32[11][k] * src[11 * line] + g_t32[13][k] * src[13 * line] + g_t32[15][k] * src[15 * line] +
				g_t32[17][k] * src[17 * line] + g_t32[19][k] * src[19 * line] + g_t32[21][k] * src[21 * line] + g_t32[23][k] * src[23 * line] +
				g_t32[25][k] * src[25 * line] + g_t32[27][k] * src[27 * line] + g_t32[29][k] * src[29 * line] + g_t32[31][k] * src[31 * line];
		}

		for (k = 0; k < 8; k++)
		{
			EO[k] = g_t32[2][k] * src[2 * line] + g_t32[6][k] * src[6 * line] + g_t32[10][k] * src[10 * line] + g_t32[14][k] * src[14 * line] +
				g_t32[18][k] * src[18 * line] + g_t32[22][k] * src[22 * line] + g_t32[26][k] * src[26 * line] + g_t32[30][k] * src[30 * line];
		}

		for (k = 0; k < 4; k++)
		{
			EEO[k] = g_t32[4][k] * src[4 * line] + g_t32[12][k] * src[12 * line] + g_t32[20][k] * src[20 * line] + g_t32[28][k] * src[28 * line];
		}

		EEEO[0] = g_t32[8][0] * src[8 * line] + g_t32[24][0] * src[24 * line];
		EEEO[1] = g_t32[8][1] * src[8 * line] + g_t32[24][1] * src[24 * line];
		EEEE[0] = g_t32[0][0] * src[0] + g_t32[16][0] * src[16 * line];
		EEEE[1] = g_t32[0][1] * src[0] + g_t32[16][1] * src[16 * line];

		/* Combining even and odd terms at each hierarchy levels to calculate the final spatial domain vector */
		EEE[0] = EEEE[0] + EEEO[0];
		EEE[3] = EEEE[0] - EEEO[0];
		EEE[1] = EEEE[1] + EEEO[1];
		EEE[2] = EEEE[1] - EEEO[1];
		for (k = 0; k < 4; k++)
		{
			EE[k] = EEE[k] + EEO[k];
			EE[k + 4] = EEE[3 - k] - EEO[3 - k];
		}

		for (k = 0; k < 8; k++)
		{
			E[k] = EE[k] + EO[k];
			E[k + 8] = EE[7 - k] - EO[7 - k];
		}

		for (k = 0; k < 16; k++)
		{
			dst[k] = (int16_t)x265_clip3(-32768, 32767, (E[k] + O[k] + add) >> shift);
			dst[k + 16] = (int16_t)x265_clip3(-32768, 32767, (E[15 - k] - O[15 - k] + add) >> shift);
		}

		src++;
		dst += 32;
	}
}

static void partialButterfly4(const int16_t* src, int16_t* dst, int shift, int line)
{
	int j;
	int E[2], O[2];
	int add = 1 << (shift - 1);

	for (j = 0; j < line; j++)
	{
		/* E and O */
		E[0] = src[0] + src[3];
		O[0] = src[0] - src[3];
		E[1] = src[1] + src[2];
		O[1] = src[1] - src[2];

		dst[0] = (int16_t)((g_t4[0][0] * E[0] + g_t4[0][1] * E[1] + add) >> shift);
		dst[2 * line] = (int16_t)((g_t4[2][0] * E[0] + g_t4[2][1] * E[1] + add) >> shift);
		dst[line] = (int16_t)((g_t4[1][0] * O[0] + g_t4[1][1] * O[1] + add) >> shift);
		dst[3 * line] = (int16_t)((g_t4[3][0] * O[0] + g_t4[3][1] * O[1] + add) >> shift);

		src += 4;
		dst++;
	}
}

static void basic(const int16_t* src, int16_t* dst, int shift1, int shift2, int line,FTYPE type){
	int16_t* mid = new int16_t[line * line];
	for (int i = 0; i < line; i++){
		for (int j = 0; j < line; j++){
			int sum = 0;
			for (int k = 0; k < line; k++){
				switch (type)
				{
				case DST:
					sum += g_ta[i][k] * src[k*line + j];
					break;
				case IDST:
					sum += g_ta[k][i] * src[k*line + j];
					break;
				case DCT4:
					sum += g_t4[i][k] * src[k*line + j];
					break;
				case IDCT4:
					sum += g_t4[k][i] * src[k*line + j];
					break;
				case DCT8:
					sum += g_t8[i][k] * src[k*line + j];
					break;
				case IDCT8:
					sum += g_t8[k][i] * src[k*line + j];
					break;
				case DCT16:
					sum += g_t16[i][k] * src[k*line + j];
					break;
				case IDCT16:
					sum += g_t16[k][i] * src[k*line + j];
					break;
				case DCT32:
					sum += g_t32[i][k] * src[k*line + j];
					break;
				case IDCT32:
					sum += g_t32[k][i] * src[k*line + j];
					break;
				}
			}
			sum >>= shift1;
			mid[i * line + j] = sum;
		}
	}

	for (int i = 0; i < line; i++){
		for (int j = 0; j < line; j++){
			int sum = 0;
			for (int k = 0; k < line; k++){
				switch (type)
				{
				case DST:
					sum += mid[i*line + k] * g_ta[j][k];
					break;
				case IDST:
					sum += mid[i*line + k] * g_ta[k][j];
					break;
				case DCT4:
					sum += mid[i*line + k] * g_t4[j][k];
					break;
				case IDCT4:
					sum += mid[i*line + k] * g_t4[k][j];
					break;
				case DCT8:
					sum += mid[i*line + k] * g_t8[j][k];
					break;
				case IDCT8:
					sum += mid[i*line + k] * g_t8[k][j];
					break;
				case DCT16:
					sum += mid[i*line + k] * g_t16[j][k];
					break;
				case IDCT16:
					sum += mid[i*line + k] * g_t16[k][j];
					break;
				case DCT32:
					sum += mid[i*line + k] * g_t32[j][k];
					break;
				case IDCT32:
					sum += mid[i*line + k] * g_t32[k][j];
					break;
				}
			}
			sum >>= shift2;
			dst[i * line + j] = sum;
		}
	}

}

int16_t* dst4_c(const int16_t* src, int16_t* dst, PTYPE PROC, int batch)
{
	const int shift_1st = 1 + X265_DEPTH - 8;
	const int shift_2nd = 8;

	ALIGN_VAR_32(int16_t, coef[4 * 4]);
	ALIGN_VAR_32(int16_t, block[4 * 4]);
	if (PROC != GPUAtomic)
	for (int i = 0; i < 4; i++)
	{
		memcpy(block, src, batch * 4 * 4 * sizeof(int16_t));
	}

	switch (PROC)
	{
	case GPUPlain:
		gpuTransformPlain<DST>(block, dst, shift_1st, shift_2nd, 4);
		break;
	case GPUMemShared:
		gpuTransformShared<DST>(block, dst, shift_1st, shift_2nd, 4);
		break;
	case GPUMinMul:
		gpuLessMulTransform<DST>(block, dst, shift_1st, shift_2nd, 4);
		break;
	case GPUOneStep:
		gpuTransform1Step<DST>(block, dst, shift_1st, shift_2nd, 4);

		break;
	case GPUAtomic:
		gpuTransform1StepBatch<DST>(block, dst, shift_1st, shift_2nd, 4, batch);
		break;
	case ASM:
		X265_NS_dst4_avx2(block, dst, 4);
		break;
	case CPU:
		fastForwardDst(block, coef, shift_1st);
		fastForwardDst(coef, dst, shift_2nd);
		break;
	default:
		basic(block, dst, shift_1st, shift_2nd, 4, DST);
		break;
	}
	return dst;
}

int16_t* dct4_c(const int16_t* src, int16_t* dst, PTYPE PROC, int batch)
{
	const int shift_1st = 1 + X265_DEPTH - 8;
	const int shift_2nd = 8;

	ALIGN_VAR_32(int16_t, coef[4 * 4]);
	ALIGN_VAR_32(int16_t, block[4 * 4]);
	if (PROC != GPUAtomic)
	for (int i = 0; i < 4; i++)
	{
		memcpy(block, src, 4 * 4 * sizeof(int16_t));
	}

	switch (PROC)
	{
	case GPUPlain:
		gpuTransformPlain<DCT4>(block, dst, shift_1st, shift_2nd, 4);
		break;
	case GPUMemShared:
		gpuTransformShared<DCT4>(block, dst, shift_1st, shift_2nd, 4);
		break;
	case GPUMinMul:
		gpuLessMulTransform<DCT4>(block, dst, shift_1st, shift_2nd, 4);
		break;
	case GPUOneStep:
		gpuTransform1Step<DCT4>(block, dst, shift_1st, shift_2nd, 4);

		break;
	case GPUAtomic:
		gpuTransform1StepBatch<DCT4>(src, dst, shift_1st, shift_2nd, 4,batch);
		break;
	case ASM:
		X265_NS_dct4_avx2(block, dst, 4);
		break;
	case CPU:
		partialButterfly4(block, coef, shift_1st, 4);
		partialButterfly4(coef, dst, shift_2nd, 4);
		break;
	default:
		basic(block, dst, shift_1st, shift_2nd, 4, DCT4);
		break;
	}
	return dst;
}

int16_t* dct8_c(const int16_t* src, int16_t* dst, PTYPE PROC, int batch)
{
	const int shift_1st = 2 + X265_DEPTH - 8;
	const int shift_2nd = 9;

	ALIGN_VAR_32(int16_t, coef[8 * 8]);
	ALIGN_VAR_32(int16_t, block[8 * 8]);
	if (PROC != GPUAtomic)
	for (int i = 0; i < 8; i++)
	{
		memcpy(block, src, 8 * 8 * sizeof(int16_t));
	}

	switch (PROC)
	{
	case GPUPlain:
		gpuTransformPlain<DCT8>(block, dst, shift_1st, shift_2nd, 8);
		break;
	case GPUMemShared:
		gpuTransformShared<DCT8>(block, dst, shift_1st, shift_2nd, 8);
		break;
	case GPUMinMul:
		gpuLessMulTransform<DCT8>(block, dst, shift_1st, shift_2nd, 8);
		break;
	case GPUOneStep:
		gpuTransform1Step<DCT8>(block, dst, shift_1st, shift_2nd, 8);

		break;
	case GPUAtomic:
		gpuTransform1StepBatch<DCT8>(src, dst, shift_1st, shift_2nd, 8,batch);
		break;
	case ASM:
		X265_NS_dct8_avx2(block, dst, 8);
		break;
	case CPU:
		partialButterfly8(block, coef, shift_1st, 8);
		partialButterfly8(coef, dst, shift_2nd, 8);
		break;
	default:
		basic(block, dst, shift_1st, shift_2nd, 8, DCT8);
		break;
	}
	return dst;
}

int16_t* dct16_c(const int16_t* src, int16_t* dst, PTYPE PROC, int batch)
{
	const int shift_1st = 3 + X265_DEPTH - 8;
	const int shift_2nd = 10;

	ALIGN_VAR_32(int16_t, coef[16 * 16]);
	ALIGN_VAR_32(int16_t, block[16 * 16]);
	if (PROC != GPUAtomic)
	for (int i = 0; i < 16; i++)
	{
		memcpy(block, src,16 * 16 * sizeof(int16_t));
	}

	switch (PROC)
	{
	case GPUPlain:
		gpuTransformPlain<DCT16>(block, dst, shift_1st, shift_2nd, 16);
		break;
	case GPUMemShared:
		gpuTransformShared<DCT16>(block, dst, shift_1st, shift_2nd, 16);
		break;
	case GPUMinMul:
		gpuLessMulTransform<DCT16>(block, dst, shift_1st, shift_2nd, 16);
		break;
	case GPUOneStep:
		gpuTransform1Step<DCT16>(block, dst, shift_1st, shift_2nd, 16);

		break;
	case GPUAtomic:
		gpuTransform1StepBatch<DCT16>(src, dst, shift_1st, shift_2nd, 16,batch);
		break;
	case ASM:
		X265_NS_dct16_avx2(block, dst, 16);
		break;
	case CPU:
		partialButterfly16(block, coef, shift_1st, 16);
		partialButterfly16(coef, dst, shift_2nd, 16);
		break;
	default:
		basic(block, dst, shift_1st, shift_2nd, 16, DCT16);
		break;
	}
	return dst;
}

int16_t* dct32_c(const int16_t* src, int16_t* dst, PTYPE PROC, int batch)
{
	const int shift_1st = 4 + X265_DEPTH - 8;
	const int shift_2nd = 11;

	ALIGN_VAR_32(int16_t, coef[32 * 32]);
	ALIGN_VAR_32(int16_t, block[32 * 32]);
	if (PROC != GPUAtomic)
	for (int i = 0; i < 32; i++)
	{
		memcpy(block, src, 32 * 32 * sizeof(int16_t));
	}

	switch (PROC)
	{
	case GPUPlain:
		gpuTransformPlain<DCT32>(block, dst, shift_1st, shift_2nd, 32);
		break;
	case GPUMemShared:
		gpuTransformShared<DCT32>(block, dst, shift_1st, shift_2nd, 32);
		break;
	case GPUMinMul:
		gpuLessMulTransform<DCT32>(block, dst, shift_1st, shift_2nd, 32);
		break;
	case GPUOneStep:
		gpuTransform1Step<DCT32>(block, dst, shift_1st, shift_2nd, 32);
		break;
	case GPUAtomic:
		gpuTransform1StepBatch<DCT32>(src, dst, shift_1st, shift_2nd, 32,batch);
		break;
	case ASM:
		X265_NS_dct32_avx2(block, dst, 32);
		break;
	case CPU:
		partialButterfly32(block, coef, shift_1st, 32);
		partialButterfly32(coef, dst, shift_2nd, 32);
		break;
	default:
		basic(block, dst, shift_1st, shift_2nd, 32, DCT32);
		break;
	}
	return dst;
}

int16_t* idst4_c(const int16_t* src, int16_t* dst, PTYPE PROC, int batch)
{
	const int shift_1st = 7;
	const int shift_2nd = 12 - (X265_DEPTH - 8);

	ALIGN_VAR_32(int16_t, coef[4 * 4]);
	ALIGN_VAR_32(int16_t, block[4 * 4]);

	switch (PROC)
	{
	case GPUPlain:
		gpuTransformPlain<IDST>(src, block, shift_1st, shift_2nd, 4);
		break;
	case GPUMemShared:
		gpuTransformShared<IDST>(src, block, shift_1st, shift_2nd, 4);
		break;
	case GPUMinMul:
		gpuLessMulTransform<IDST>(src, block, shift_1st, shift_2nd, 4);
		break;
	case GPUOneStep:
		gpuTransform1Step<IDST>(src, block, shift_1st, shift_2nd, 4);
		break;
	case GPUAtomic:
		gpuTransform1StepBatch<IDST>(src, dst, shift_1st, shift_2nd, 4,batch);
		break;
	case ASM:
		X265_NS_idst4_avx2(src, block, 4);
		break;
	case CPU:
		inversedst(src, coef, shift_1st); // Forward DST BY FAST ALGORITHM, block input, coef output
		inversedst(coef, block, shift_2nd); // Forward DST BY FAST ALGORITHM, coef input, coeff output
		break;
	default:
		basic(src, block, shift_1st, shift_2nd, 4, IDST);
		break;
	}
	if (PROC != GPUAtomic)
	for (int i = 0; i < 4; i++)
	{
		memcpy(dst, block, 4 * 4 * sizeof(int16_t));
	}
	return dst;
}

int16_t* idct4_c(const int16_t* src, int16_t* dst, PTYPE PROC,int batch)
{
	const int shift_1st = 7;
	const int shift_2nd = 12 - (X265_DEPTH - 8);

	ALIGN_VAR_32(int16_t, coef[4 * 4]);
	ALIGN_VAR_32(int16_t, block[4 * 4]);

	switch (PROC)
	{
	case GPUPlain:
		gpuTransformPlain<IDCT4>(src, block, shift_1st, shift_2nd, 4);
		break;
	case GPUMemShared:
		gpuTransformShared<IDCT4>(src, block, shift_1st, shift_2nd, 4);
		break;
	case GPUMinMul:
		gpuLessMulTransform<IDCT4>(src, block, shift_1st, shift_2nd, 4);
		break;
	case GPUOneStep:
		gpuTransform1Step<IDCT4>(src, block, shift_1st, shift_2nd, 4);
		break;
	case GPUAtomic:
		gpuTransform1StepBatch<IDCT4>(src, dst, shift_1st, shift_2nd, 4,batch);
		break;
	case ASM:
		X265_NS_idct4_avx2(src, block, 4);
		break;
	case CPU:
		partialButterflyInverse4(src, coef, shift_1st, 4); // Forward DST BY FAST ALGORITHM, block input, coef output
		partialButterflyInverse4(coef, block, shift_2nd, 4); // Forward DST BY FAST ALGORITHM, coef input, coeff output
		break;
	default:
		basic(src, block, shift_1st, shift_2nd, 4, IDCT4);
		break;
	}
	if (PROC != GPUAtomic)
	for (int i = 0; i < 4; i++)
	{
		memcpy(dst, block, 4 * 4 * sizeof(int16_t));
	}
	return dst;
}

int16_t* idct8_c(const int16_t* src, int16_t* dst, PTYPE PROC, int batch)
{
	const int shift_1st = 7;
	const int shift_2nd = 12 - (X265_DEPTH - 8);

	ALIGN_VAR_32(int16_t, coef[8 * 8]);
	ALIGN_VAR_32(int16_t, block[8 * 8]);

	switch (PROC)
	{
	case GPUPlain:
		gpuTransformPlain<IDCT8>(src, block, shift_1st, shift_2nd, 8);
		break;
	case GPUMemShared:
		gpuTransformShared<IDCT8>(src, block, shift_1st, shift_2nd, 8);
		break;
	case GPUMinMul:
		gpuLessMulTransform<IDCT8>(src, block, shift_1st, shift_2nd, 8);
		break;
	case GPUOneStep:
		gpuTransform1Step<IDCT8>(src, block, shift_1st, shift_2nd, 8);
		break;
	case GPUAtomic:
		gpuTransform1StepBatch<IDCT8>(src, dst, shift_1st, shift_2nd, 8,batch);
		break;
	case ASM:
		X265_NS_idct8_avx2(src, block, 8);
		break;
	case CPU:
		partialButterflyInverse8(src, coef, shift_1st, 8);
		partialButterflyInverse8(coef, block, shift_2nd, 8);
		break;
	default:
		basic(src, block, shift_1st, shift_2nd, 8, IDCT8);
		break;
	}
	if (PROC != GPUAtomic)
	for (int i = 0; i < 8; i++)
	{
		memcpy(dst, block, 8 * 8 * sizeof(int16_t));
	}
	return dst;
}

int16_t* idct16_c(const int16_t* src, int16_t* dst, PTYPE PROC, int batch)
{
	const int shift_1st = 7;
	const int shift_2nd = 12 - (X265_DEPTH - 8);

	ALIGN_VAR_32(int16_t, coef[16 * 16]);
	ALIGN_VAR_32(int16_t, block[16 * 16]);

	switch (PROC)
	{
	case GPUPlain:
		gpuTransformPlain<IDCT16>(src, block, shift_1st, shift_2nd, 16);
		break;
	case GPUMemShared:
		gpuTransformShared<IDCT16>(src, block, shift_1st, shift_2nd, 16);
		break;
	case GPUMinMul:
		gpuLessMulTransform<IDCT16>(src, block, shift_1st, shift_2nd, 16);
		break;
	case GPUOneStep:
		gpuTransform1Step<IDCT16>(src, block, shift_1st, shift_2nd, 16);
		break;
	case GPUAtomic:
		gpuTransform1StepBatch<IDCT16>(src, dst, shift_1st, shift_2nd, 16,batch);
		break;
	case ASM:
		X265_NS_idct16_avx2(src, block, 16);
		break;
	case CPU:
		partialButterflyInverse16(src, coef, shift_1st, 16);
		partialButterflyInverse16(coef, block, shift_2nd, 16);
		break;
	default:
		basic(src, block, shift_1st, shift_2nd, 16, IDCT16);
		break;
	}
	if (PROC != GPUAtomic)
	for (int i = 0; i < 16; i++)
	{
		memcpy(dst, block, 16 * 16 * sizeof(int16_t));
	}
	return dst;
	
}

int16_t* idct32_c(const int16_t* src, int16_t* dst, PTYPE PROC,int batch)
{
	const int shift_1st = 7;
	const int shift_2nd = 12 - (X265_DEPTH - 8);

	ALIGN_VAR_32(int16_t, coef[32 * 32]);
	ALIGN_VAR_32(int16_t, block[32 * 32]);

	switch (PROC)
	{
	case GPUPlain:
		gpuTransformPlain<IDCT32>(src, block, shift_1st, shift_2nd, 32);
		break;
	case GPUMemShared:
		gpuTransformShared<IDCT32>(src, block, shift_1st, shift_2nd, 32);
		break;
	case GPUMinMul:
		gpuLessMulTransform<IDCT32>(src, block, shift_1st, shift_2nd, 32);
		break;
	case GPUOneStep:
		gpuTransform1Step<IDCT32>(src, block, shift_1st, shift_2nd, 32);
		break;
	case GPUAtomic:
		gpuTransform1StepBatch<IDCT32>(src, dst, shift_1st, shift_2nd, 32,batch);
		break;
	case ASM:
		X265_NS_idct32_avx2(src, block, 32);
		break;
	case CPU:
		partialButterflyInverse32(src, coef, shift_1st, 32);
		partialButterflyInverse32(coef, block, shift_2nd, 32);
		break;
	default:
		basic(src, block, shift_1st, shift_2nd, 32, IDCT32);
		break;
	}
	if (PROC != GPUAtomic)
	for (int i = 0; i < 32; i++)
	{
		memcpy(dst, block, 32 * 32 * sizeof(int16_t));
	}
	return dst;
}