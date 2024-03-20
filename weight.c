#include "./weight.h"

Tensor weight_initialize(
  Tensor *tensor,
  int32 in_features,
  int32 out_features,
  Method method
) {
  Tensor weight = initialize(in_features, out_features);

  switch (method) {
    case ZEROS:
      weight = zeros(in_features, out_features);
      break;
    case ONES:
      weight = ones(in_features, out_features);
      break;
    case CONSTANT:
    case RAND_LIKE:
    case TRUNCN:
    case VARSCALE:
    case ORTHOGONAL:
    case ID:
    default:
    case RANDN:
      weight = randn(in_features, out_features);
      break;
  }

  return weight;
}

int main() {
  srand(time(NULL));
  Tensor t1 = randn(3, 1);
  Tensor w = weight_initialize(&t1, 3, 4, ZEROS);

  printf("(%zu, %zu)\n", t1.col, t1.row);
  printf("(%zu, %zu)\n", w.col, w.row);
  print_tensor(&t1);
  print_tensor(&w);

  free(t1.value);
  free(w.value);

  return 0;
}