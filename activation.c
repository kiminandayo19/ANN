#include "./activation.h"

Tensor *sigmoid_a(
  Tensor *y
) {
  Tensor *a = initialize(y->ndim, y->shape);
  for (int32 i=0; i<get_size(y->ndim, y->shape); i++) {
    a->value[i] = (1.0 / (1.0 + exp(-(y->value[i]))));
  }
  return a;
}

Tensor *relu_a(
  Tensor *y
) {
  Tensor *a = initialize(y->ndim, y->shape);
  for (int32 i=0; i<get_size(y->ndim, y->shape); i++) {
    a->value[i] = MAX(0.0, y->value[i]);
  }
  return a;
}

Tensor *leaky_relu_a(
  Tensor *y
) {
  Tensor *a = initialize(y->ndim, y->shape);
  for (int32 i=0; i<get_size(y->ndim, y->shape); i++) {
    if (y->value[i] < 0.0) {
      a->value[i] = 0.01 * y->value[i];
    } else {
      a->value[i] = y->value[i];
    }
  }
  return a;
}

Tensor *parametric_relu_a(
  Tensor *y,
  float64 k
) {
  Tensor *a = initialize(y->ndim, y->shape);
  for (int32 i=0; i<get_size(y->ndim, y->shape); i++) {
    if (y->value[i] < 0.0) {
      a->value[i] = k * y->value[i];
    } else {
      a->value[i] = y->value[i];
    }
  }
  return a;
}

void print_t(
  Tensor *t
) {
  for (int32 i=0; i<get_size(t->ndim, t->shape); i++) {
    printf("%f ", t->value[i]);
  }
  printf("\n");
}