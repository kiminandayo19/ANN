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
#define LEN(x) sizeof(x) / sizeof(x[0])

typedef float float32;
typedef double float64;
typedef int32_t int32;
typedef int64_t int64;

typedef struct {
  size_t ndim;
  int64 *shape;
  char dtype[16];
  float64 *value;
} Tensor;

Tensor initialize(size_t ndim, int64 *shape);
Tensor zeros(size_t ndim, int64 *shape);
Tensor ones(size_t ndim, int64 *shape);
Tensor randn(size_t ndim, int64 *shape);
Tensor cust_randn(
  size_t ndim,
  int64 *shape,
  float64 mu,
  float64 stdev
);
Tensor add(Tensor *a, Tensor *b);
Tensor substract(Tensor *a, Tensor *b);
Tensor scalar_mul(float64 k, Tensor *in);
Tensor element_mul(Tensor *a, Tensor *b);
Tensor matrix_vector_mul(Tensor *w, Tensor *x);

int64 get_size(size_t ndim, int64 *shape);
void assign_scalar(Tensor *tensor, float64 scalar);
void assign_values(Tensor *tensor, float64 *values);
void print_tensor(Tensor *tenor);
void free_tensor(Tensor *tensor);

#endif
