#include "./datatype.h"
#include "./tensor.h"
#include "./ops.h"
#include "./weight.h"

#ifndef ACTIVATION_H
#define ACTIVATION_H

Tensor *sigmoid_a(
  Tensor *y
);
Tensor *tanh_a(
  Tensor *y
);
Tensor *relu_a(
  Tensor *y
);
Tensor *leaky_relu_a(
  Tensor *y
);
Tensor *parametric_relu_a(
  Tensor *y,
  float64 k
);
void print_t(Tensor *t);

#endif