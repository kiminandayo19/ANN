#include "./init.h"

#ifndef LOSS_H
#define LOSS_H

float64 MSE_loss(
  Tensor *y,
  Tensor *y_hat
);

float64 L1_loss(
  Tensor *y,
  Tensor *y_hat
);

#endif