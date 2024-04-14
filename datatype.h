#include <stdio.h>

#ifndef DATATYPE_H
#define DATATYPE_H

typedef long long int int64;
typedef int int32;
typedef double float64;
typedef float float32;

typedef struct {
  size_t ndim;
  int64 *shape;
  char dtype[16];
  float64 *value;
} Tensor;

#endif