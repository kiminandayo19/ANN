#include "./tensor.h"

int64 get_size(
  size_t _ndim,
  int64 *_shape
) {
  int64 _size = 1;
  for (int32 i=0; i<_ndim; i++) {
    _size *= _shape[i];
  }
  return _size;
}

Tensor *initialize(
  size_t _ndim,
  int64 *_shape
) {
  Tensor *init = (Tensor *)malloc(sizeof(Tensor));
  if (init == NULL) {
    fprintf(stderr, "Failed to allocate tensor memory\n");
    exit(1);
  }

  init->ndim = _ndim;
  init->shape = (int64 *)malloc(_ndim * sizeof(int64));
  if (init->shape == NULL) {
    fprintf(stderr, "Failed to allocate shape memory\n");
    exit(1);
  }
  for (int32 i=0; i<_ndim; i++)
    init->shape[i] = _shape[i];

  init->value = (float64*)malloc(get_size(_ndim, _shape) * sizeof(float64));
  if (init->value == NULL) {
    fprintf(stderr, "Failed to allocate value memory\n");
    exit(1);
  }
  return init;
}

Tensor *zeros(
  size_t _ndim,
  int64 *_shape
) {
  return initialize(_ndim, _shape);
}

Tensor *ones(
  size_t _ndim,
  int64 *_shape
) {
  Tensor *one = initialize(_ndim, _shape);
  for (int32 i=0; i<get_size(_ndim, _shape); i++)
    one->value[i] = 1.0;
  return one;
}

Tensor *randn(
  size_t _ndim,
  int64 *_shape
) {
  Tensor *random = initialize(_ndim, _shape);
  for (int32 i=0; i<get_size(_ndim, _shape); i++)
    random->value[i] = gaussian_sampling(0.0f, 1.0f);
  return random;
}

Tensor *sample(
  size_t _ndim,
  int64 *_shape,
  float64 _mean,
  float64 _std
) {
  Tensor *random = initialize(_ndim, _shape);
  for (int32 i=0; i<get_size(_ndim, _shape); i++)
    random->value[i] = gaussian_sampling(_mean, _std);
  return random;
}

float64 gaussian_sampling(float64 mu, float64 stdev)
{
  static float64 z0, z1;
  static float64 threshold = 1e-7;
  static int32 generate;
  generate = !generate;

  if (!generate) return z1 * stdev + mu;

  float64 u1, u2;
  do {
    u1 = rand() * (1.0 / RAND_MAX);
    u2 = rand() * (1.0 / RAND_MAX);
  } while (u1 <= threshold);

  float64 R = sqrt(-2.0 * log(u1));
  float64 theta = 2.0 * M_PI * u2;
  z0 = R * cos(theta);
  z1 = R * sin(theta);
  return z0 * stdev + mu;
}

void free_tensor(
  Tensor *tensor
) {
  free(tensor->shape);
  free(tensor->value);
  free(tensor);
}