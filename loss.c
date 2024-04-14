#include "./loss.h"

float64 MSE_loss(
  Tensor *y,
  Tensor *y_hat
) {
  float64 _sum = 0.0f;
  int64 _size = get_size(y->ndim, y->shape);
  for (int32 i=0; i<_size; i++) {
    _sum += ((y->value[i]) - (y_hat->value[i])) * ((y->value[i]) - (y_hat->value[i]));
  }
  return (_sum / _size);
}

float64 L1_loss(
  Tensor *y,
  Tensor *y_hat
) {
  float64 _sum = 0.0f;
  int32 _size = get_size(y->ndim, y->shape);
  for (int32 i=0; i<_size; i++) {
    _sum += fabs((y->value[i]) - (y_hat->value[i]));
  }
  return _sum;
}