#include "./bias.h"

Tensor *zero_b(
  Tensor *y
) {
  Tensor *b = zeros(y->ndim, y->shape);
  return b;
}