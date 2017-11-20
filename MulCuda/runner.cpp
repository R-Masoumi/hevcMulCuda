#include "runner.h"
inline std::ostream & operator<<(std::ostream & Str, PTYPE V) {
	switch (V)
	{
	case GPUPlain:
		return Str << "GPU Plain";
		break;
	case GPUMemShared:
		return Str << "GPU Shared";
		break;
	case GPUMinMul:
		return Str << "GPU Min Mult";
		break;
	case GPUOneStep:
		return Str << "GPU 1Step";

		break;
	case GPUAtomic:
		return Str << "GPU Batch";
		break;
	case ASM:
		return Str << "AVX2";
		break;
	case CPU:
		return Str << "CPU";
		break;
	default:
		return Str << "Basic CPU";
		break;
	}
}
inline std::ostream & operator<<(std::ostream & Str, FTYPE V) {
	switch (V)
	{
	case DST:
		return Str << "DST";
		break;
	case DCT4:
		return Str << "DCT4";
		break;
	case DCT8:
		return Str << "DCT8";
		break;
	case DCT16:
		return Str << "DCT16";
		break;
	case DCT32:
		return Str << "DCT32";
		break;
	case IDST:
		return Str << "Inverse DST";
		break;
	case IDCT4:
		return Str << "Inverse DCT4";
		break;
	case IDCT8:
		return Str << "Inverse DCT8";
		break;
	case IDCT16:
		return Str << "Inverse DCT16";
		break;
	case IDCT32:
		return Str << "Inverse DCT32";
		break;
	default:
		return Str << "Unkown Transformation";
		break;
	}
}


void print_matrix(const int16_t *A, int n, FTYPE funcType, PTYPE procType, int batch) {
	std::cout << funcType << " transform "
		<< procType << std::endl;
	for (int k = 0; k < batch; k++){
		for (int i = 0; i < n; i++){
			for (int j = 0; j < n; j++){
				std::cout << A[k*n*n + i * n + j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void print_matrix(const int16_t *A, int n,int batch) {
	std::cout << " source "<< std::endl;
	for (int k = 0; k < batch; k++){
		for (int i = 0; i < n; i++){
			for (int j = 0; j < n; j++){
				std::cout << A[k*n*n + i * n + j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void testRun(int16_t* src, int16_t* dst){
	print_matrix(src, 4);
	for (int proc = ASM; proc <= GPUOneStep; proc++){
		for (int func = DST; func <= IDCT32; func++){
			testRun(static_cast<FTYPE>(func), static_cast<PTYPE>(proc), src, dst);
		}
	}
}

void testRun(FTYPE funcType, int16_t* src, int16_t* dst){
	for (int proc = ASM; proc <= GPUOneStep; proc++){
		testRun(funcType, static_cast<PTYPE>(proc), src, dst);
	}
}

void testRun(PTYPE procType, int16_t* src, int16_t* dst){
	print_matrix(src, 4);
	for (int func = DST; func <= IDCT32; func++){
		testRun(static_cast<FTYPE>(func), procType, src, dst);
	}
}

void testRun(FTYPE funcType,PTYPE procType, int16_t* src, int16_t* dst,int batch){
	switch (funcType)
	{
		case DST:
			dst4_c(src, dst, procType, batch);
			print_matrix(dst, 4, funcType, procType, batch);
			break;
		case DCT4:
			dct4_c(src, dst, procType, batch);
			print_matrix(dst, 4, funcType, procType, batch);
			break;
		case DCT8:
			dct8_c(src, dst, procType, batch);
			print_matrix(dst, 8, funcType, procType, batch);
			break;
		case DCT16:
			dct16_c(src, dst, procType, batch);
			print_matrix(dst, 16, funcType, procType, batch);
			break;
		case DCT32:
			dct32_c(src, dst, procType, batch);
			print_matrix(dst, 32, funcType, procType, batch);
			break;
		case IDST:
			idst4_c(src, dst, procType, batch);
			print_matrix(dst, 4, funcType, procType, batch);
			break;
		case IDCT4:
			idct4_c(src, dst, procType, batch);
			print_matrix(dst, 4, funcType, procType, batch);
			break;
		case IDCT8:
			idct8_c(src, dst, procType, batch);
			print_matrix(dst, 8, funcType, procType, batch);
			break;
		case IDCT16:
			idct16_c(src, dst, procType, batch);
			print_matrix(dst, 16, funcType, procType, batch);
			break;
		case IDCT32:
			idct32_c(src, dst, procType, batch);
			print_matrix(dst, 32, funcType, procType, batch);
			break;
	}
}
