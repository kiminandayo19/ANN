#include "./ops.h"

Tensor *add_t(
  Tensor *a,
  Tensor *b
) {
  size_t _ndim = a->ndim;
  int64 *_shape = a->shape;
  Tensor *c = initialize(_ndim, _shape);
  for (int32 i=0; i<get_size(_ndim, _shape); i++)
    c->value[i] = a->value[i] + b->value[i];
  return c;
}

Tensor *subs_t(
  Tensor *a,
  Tensor *b
) {
  size_t _ndim = a->ndim;
  int64 *_shape = a->shape;
  Tensor *c = initialize(_ndim, _shape);
  for (int32 i=0; i<get_size(_ndim, _shape); i++)
    c->value[i] = a->value[i] - b->value[i];
  return c;
}

Tensor *smul_t(
  float64 c,
  Tensor *a
) {
  size_t _ndim = a->ndim;
  int64 *_shape = a->shape;
  Tensor *k = initialize(_ndim, _shape);
  for (int32 i=0; i<get_size(_ndim, _shape); i++)
    k->value[i] = c * a->value[i];
  return k;
}

Tensor *elmul_t(
  Tensor *a,
  Tensor *b
) {
  size_t _ndim = a->ndim;
  int64 *_shape = a->shape;
  Tensor *c = initialize(_ndim, _shape);
  for (int32 i=0; i<get_size(_ndim, _shape); i++)
    c->value[i] = a->value[i] * b->value[i];
  return c;
}

Tensor *dotprod_t(
  Tensor *a,
  Tensor *b
) {
  /* Work only for 2d and 1d tensor */
  size_t _ndim = 1;
  int64 *_shape = {&a->shape[0]};
  Tensor *c = initialize(_ndim, _shape);
  for (int32 i=0; i<a->shape[0]; i++) {
    float64 sum = 0.0f;
    for (int32 j=0; j<a->shape[1]; j++) {
      sum += a->value[i * a->shape[1] + j] * b->value[j];
    }
    c->value[i] = sum;
  }
  return c;
}

float64 mean(
  Tensor *y
) {
  float64 _sum = 0.0f;
  int64 _size = get_size(y->ndim, y->shape);

  for (int32 i=0; i<_size; i++) {
    _sum += y->value[i];
  }
  return (_sum) / (_size);
}