#include "./linear.h"

int main() {
  srand(time(NULL));

  size_t _ndim = 1;
  int64 _shape[] = {4};
  int64 _in = 4, _out = 6;

  Tensor *x = randn(_ndim, _shape);
  Tensor *ff = Linear(x, _in, _out);

  print_t(x);
  print_t(ff);

  free_tensor(x);
  free_tensor(ff);

  return 0;
}