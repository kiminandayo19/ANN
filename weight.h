#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdarg.h> /* We use variadic fn for handling default*/
#include <stdbool.h>
#include "./tensor.h"

#ifndef WEIGHT_H
#define WEIGHT_H

typedef enum {
  ZEROS,
  ONES,
  CONSTANT,
  RANDN,
  RAND_LIKE,
  TRUNCN,
  VARSCALE,
  ORTHOGONAL,
  ID
} Method;

Tensor weight_initialize(
  Tensor *tensor,
  int32 in_features,
  int32 out_features,
  Method method
);
Tensor Linear(int32 in_features, int32 out_features, bool use_bias, ...);

#endif