#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <stdarg.h>
#include "./datatype.h"

#ifndef TENSOR_H
#define TENSOR_H
#define MAX(a, b) ((a) > (b) ? a : b)
#define MIN(a, b) ((a) < (b) ? a : b)

int64 get_size(
  size_t _ndim,
  int64 *_shape
);

Tensor *initialize(
  size_t _ndim,
  int64 *_shape
);
Tensor *zeros(
  size_t _ndim,
  int64 *_shape
);
Tensor *ones(
  size_t _ndim,
  int64 *_shape
);
Tensor *randn(
  size_t _ndim,
  int64 *_shape
);
Tensor *sample(
  size_t _ndim,
  int64 *_shape,
  float64 _mean,
  float64 _std
);

float64 gaussian_sampling(
  float64 mu,
  float64 std
);

void free_tensor(Tensor *tensor);

#endif