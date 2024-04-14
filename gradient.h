#include "./init.h"

#ifndef GRADIENT_H
#define GRADIENT_H

Tensor *weight_grad(
  Tensor *x,
  Tensor *y,
  Tensor *y_hat
);

Tensor *bias_grad(
  Tensor *y,
  Tensor *y_hat
);

Tensor *compute_grad(
  Tensor *y
);

Tensor *SGD_optimizer(
  Tensor *W,
  Tensor *b,
  Tensor *x,
  Tensor *y_hat,
  float64 lr,
  float64 momentum,
  float64 dampening,
  float64 weight_decay
);

#endif