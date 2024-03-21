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
  KAIMING_HE,
  LECUN,
  XAVIER,
  ID
} Method;

typedef enum {
  LIN,
  SIG,
  TANH,
  RELU,
  LRELU,
  PRELU
} Activation;

Tensor weight_init(
  int64 in_features,
  int64 out_features,
  Method method,
  float64 k
);
Tensor zeros_w(
  int64 in_features,
  int64 out_features
);
Tensor ones_w(
  int64 in_features,
  int64 out_features
);
Tensor const_w(
  int64 in_features,
  int64 out_features,
  float64 k
);
Tensor randn_w(
  int64 in_features,
  int64 out_features
);
Tensor kaiming_he(
  int64 in_features,
  int64 out_features
);
Tensor lecun(
  int64 in_features,
  int64 out_features
);
Tensor xavier(
  int64 in_features,
  int64 out_features
);
Tensor id_w(
  int64 in_feat,
  int64 out_feat
);
Tensor bias_init(
  int64 out_features
);
Tensor activation(
  Tensor *Y,
  Activation activation
);
Tensor Linear(
  int64 in_features,
  int64 out_features,
  Activation activation
);
#endif