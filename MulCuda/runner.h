#include "dct.h"
#include <limits>

void print_matrix(const int16_t *A, int n, FTYPE funcType, PTYPE procType,int batch = 1);

void print_matrix(const int16_t *A, int n, int batch = 1);

void testRun(int16_t* src, int16_t* dst);

void testRun(FTYPE funcType,int16_t* src, int16_t* dst);

void testRun(PTYPE procType, int16_t* src, int16_t* dst);

void testRun(FTYPE funcType, PTYPE procType, int16_t* src, int16_t* dst,int batch = 1);