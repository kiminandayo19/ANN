#include "./tensor.h"

/* =={Checker & Utils Method}== */

int64 get_size(size_t ndim, int64 *shape)
{
  if (ndim <= 0) {
    printf("Invalid tensor dim\n");
    exit(1);
  }
  int64 size = 1;
  for (int32 i=0; i<ndim; i++)
    size *= shape[i];
  return size;
}

static int64 is_same_shape_entry(Tensor *a, Tensor *b)
{
  int64 size = get_size(a->ndim, a->shape);
  for (int32 i=0; i<size; i++) {
    if (a->shape[i] != b->shape[i]) {
      return 0;
    }
  }
  return 1;
}

static void is_same_shape(Tensor *a, Tensor *b)
{
  if (a->ndim != b->ndim) {
    printf("Different dim\n");
    exit(1);
  }

  if ((is_same_shape_entry(a, b)) == 0) {
    printf("Different shape entry\n");
    exit(1);
  }
}

static void is_valid_matvec_mul(Tensor *a, Tensor *b)
{
  if (a->shape[1] != b->shape[0]) {
    printf("Invalid Shape\n");
    exit(1);
  }
}

void free_tensor(Tensor *tensor)
{
  free(tensor->value);
}

Tensor initialize(size_t ndim, int64 *shape)
{
  int64 size = get_size(ndim, shape);
  Tensor init = {
    .ndim = ndim,
    .shape = shape,
    .dtype = "float64",
    .value = (float64*)malloc(size * sizeof(float64)),
  };
  if (init.value == NULL) {
    printf("Failed to alocate memory\n");
    exit(1);
  }
  return init;
}

/* =={Assign Method}== */

void assign_scalar(Tensor *tensor, float64 scalar)
{
  int64 size = get_size(tensor->ndim, tensor->shape);
  memset(tensor->value, scalar, size);
}

void assign_values(Tensor *tensor, float64 *values)
{
  int64 size = get_size(tensor->ndim, tensor->shape);
  for (int32 i=0; i<size; i++)
    tensor->value[i] = values[i];
}

static float64 gaussian_sample(float64 mu, float64 stdev)
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

/* =={Init Tensor With Value Method}== */

Tensor zeros(size_t ndim, int64 *shape)
{
  Tensor tensor = initialize(ndim, shape);
  assign_scalar(&tensor, 0.0);
  return tensor;
}

Tensor ones(size_t ndim, int64 *shape)
{
  Tensor tensor = initialize(ndim, shape);
  assign_scalar(&tensor, 1.0);
  return tensor;
}

Tensor randn(size_t ndim, int64 *shape)
{
  Tensor tensor = initialize(ndim, shape);
  int64 size = get_size(ndim, shape);
  for (int32 i=0; i<size; i++) {
    tensor.value[i] = gaussian_sample(0.0, 1.0);
  }
  return tensor;
}

Tensor cust_randn(
  size_t ndim,
  int64 *shape,
  float64 mu,
  float64 stdev
) {
  Tensor tensor = initialize(ndim, shape);
  int64 size = get_size(ndim, shape);
  for (int32 i=0; i<size; i++)
    tensor.value[i] = gaussian_sample(mu, stdev);
  return tensor;
}

/* =={Tensor Manipulation Method}== */

Tensor add(Tensor *a, Tensor *b)
{
  is_same_shape(a, b);
  Tensor tensor = initialize(a->ndim, a->shape);
  int64 size = get_size(tensor.ndim, tensor.shape);
  for (int32 i=0; i<size; i++)
    tensor.value[i] = a->value[i] + b->value[i];
  return tensor;
}

Tensor substract(Tensor *a, Tensor *b)
{
  is_same_shape(a, b);
  Tensor tensor = initialize(a->ndim, a->shape);
  int64 size = get_size(tensor.ndim, tensor.shape);
  for (int32 i=0; i<size; i++)
    tensor.value[i] = a->value[i] - b->value[i];
  return tensor;
}

Tensor scalar_mul(float64 k, Tensor *in)
{
  Tensor tensor = initialize(in->ndim, in->shape);
  int64 size = get_size(tensor.ndim, tensor.shape);
  for (int32 i=0; i<size; i++)
    tensor.value[i] = (k) * (in->value[i]);
  return tensor;
}

Tensor element_mul(Tensor *a, Tensor *b)
{
  is_same_shape(a, b);
  Tensor tensor = initialize(a->ndim, a->shape);
  int64 size = get_size(tensor.ndim, tensor.shape);
  for (int32 i=0; i<size; i++)
    tensor.value[i] = a->value[i] * b->value[i];
  return tensor;
}

Tensor matrix_vector_mul(Tensor *W, Tensor *x)
{
  /* Currently only work for matrix vector op */
  is_valid_matvec_mul(W, x);
  Tensor tensor = initialize(1, &W->shape[0]);
  for (int32 i=0; i<W->shape[0]; i++) {
    float64 sum = 0.0;
    for (int32 j=0; j<W->shape[1]; j++) {
      sum += W->value[i * W->shape[1] + j] * x->value[j];
    }
    tensor.value[i] = sum;
  }
  return tensor;
}

/* =={Other}== */
void print_tensor(Tensor *tensor)
{
  printf("[");
  int64 size = get_size(tensor->ndim, tensor->shape);
  for (int32 i=0; i<size; i++) {
    printf(((i) == (size - 1)) ? "%1.4f " : "%1.4f, ", tensor->value[i]);
  }
  printf("]\n");
}