#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <stdarg.h>

#ifndef TENSOR_H
#define TENSOR_H
#define LEN(x) ((x).col)

typedef float float32;
typedef double float64;
typedef int32_t int32;
typedef int64_t int64;

typedef struct {
  size_t row;
  size_t col;
  char dtype[8];
  float32 *value;
} Tensor;

Tensor initialize(int32 len, ...);
Tensor randn(int32 len, ...);
Tensor zeros(int32 len, ...);
Tensor ones(int32 len, ...);
Tensor add(Tensor *tensor1, Tensor *tensor2);
Tensor substract(Tensor *tensor1, Tensor *tensor2);
Tensor element_wise(Tensor *tensor1, Tensor *tensor2);

float32 dot_product(Tensor *tensor1, Tensor *tensor2);
void print_tensor(Tensor *tensor);


#endif
