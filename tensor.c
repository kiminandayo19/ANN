#include "./tensor.h"

void is_valid_len(int32 len)
{
  if (len < 0) {
    printf("Tensor len cannot be less or equal zero\n");
    exit(1);
  }
}

void is_success_allocate(Tensor *tensor)
{
  if (tensor->value == NULL) {
    printf("Failed to allocate memory\n");
    exit(1);
  }
}

void is_same_length(Tensor *tensor1, Tensor *tensor2)
{
  if (tensor1->col != tensor2->col) {
    printf("Two tensor need to have same lenght\n");
    exit(1);
  }
}

Tensor initialize(int32 len, ...)
{
  va_list args;
  va_start(args, len);
  Tensor tensor;
  tensor.row = va_arg(args, size_t);
  tensor.row = (tensor.row == 0) ? 1 : tensor.row;
  va_end(args);
  tensor.col = len;
  strcpy(tensor.dtype, "float32");
  tensor.value = (float32*)malloc(len * tensor.row * sizeof(float32));
  is_success_allocate(&tensor);
  return tensor;
}

void assign(Tensor *tensor, float32 *values)
{
  for (int i=0; i<tensor->col; i++)
    tensor->value[i] = values[i];
}

double generate_random_normal(void)
{
  static float64 z0, z1;
  static int32 generate;
  float64 mean = 0.0;
  float64 stdev = 1.0;
  generate = !generate;

  if (!generate)
    return z1 * stdev + mean;

  float64 u1, u2;
  do {
    u1 = rand() * (1.0 / RAND_MAX);
    u2 = rand() * (1.0 / RAND_MAX);
  } while (u1 <= 1e-7);

  float64 R = sqrt(-2.0 * log(u1));
  float64 theta = 2.0 * M_PI * u2;
  z0 = R * cos(theta);
  z1 = R * sin(theta);
  return z0 * stdev + mean;
}

Tensor randn(int32 len, ...)
{
  va_list args;
  va_start(args, len);
  int row = va_arg(args, int);
  row = (row == 0) ? 1 : row;
  Tensor tensor = initialize(len, row);
  va_end(args);
  for (int i=0; i<tensor.col * row; i++)
    tensor.value[i] = generate_random_normal();
  return tensor;
}

Tensor zeros(int32 len, ...)
{
  va_list args;
  va_start(args, len);
  size_t row = va_arg(args, int32);
  row = (row == 0) ? 1 : row;
  Tensor tensor = initialize(len, row);
  va_end(args);

  size_t size = tensor.col * row;
  memset(tensor.value, 0.0, size);
  return tensor;
}

Tensor ones(int32 len, ...)
{
  va_list args;
  va_start(args, len);
  size_t row = va_arg(args, int32);
  Tensor tensor = initialize(len, row);
  va_end(args);

  size_t size = tensor.col * row;
  memset(tensor.value, 1.0, size);
  return tensor;
}

Tensor add(Tensor *tensor1, Tensor *tensor2)
{
  is_same_length(tensor1, tensor2);
  Tensor tensor = initialize(tensor1->col);
  for (int i=0; i<tensor.col; i++) {
    tensor.value[i] = tensor1->value[i] + tensor2->value[i];
  }
  return tensor;
}

Tensor substract(Tensor *tensor1, Tensor *tensor2)
{
  is_same_length(tensor1, tensor2);
  Tensor tensor = initialize(tensor1->col);
  for (int i=0; i<tensor.col; i++) {
    tensor.value[i] = tensor1->value[i] - tensor2->value[i];
  }
  return tensor;
}

float32 dot_product(Tensor *tensor1, Tensor *tensor2)
{
  is_same_length(tensor1, tensor2);
  float32 result = 0.0;
  for (int i=0; i<tensor1->col; i++)
    result += (tensor1->value[i] * tensor2->value[i]);
  return result;
}

Tensor element_wise(Tensor *tensor1, Tensor *tensor2)
{
  is_same_length(tensor1, tensor2);
  Tensor tensor = initialize(tensor1->col);
  for (int i=0; i<tensor.col; i++) {
    tensor.value[i] = tensor1->value[i] * tensor2->value[i];
  }
  return tensor;
}

void print_tensor(Tensor *tensor) {
  printf("[");
  for (int i=0; i<tensor->col * tensor->row; i++)
    printf((i == tensor->col - 1) ? "%1.4f " : "%1.4f, ", tensor->value[i]);
  printf("]\n");
}