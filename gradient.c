#include "./gradient.h"

Tensor *weight_grad(
  Tensor *x,
  Tensor *y,
  Tensor *y_hat
) {}

Tensor *bias_grad(
  Tensor *y,
  Tensor *y_hat
) {}

Tensor *SGD_Optimizer(
  Tensor *w,
  Tensor *b,
  Tensor *x,
  Tensor *y_hat,
  float64 lr,
  float64 momentum,
  float64 dampening,
  float64 weight_decay
) {
  // Compute grad for weight and bias
  
  // Update weight and bias
}

int main() {
  // TODO => Complete function to calculate grad for weight and bias
  // TODO => Implement SGD Optimizer
  return 0;
}