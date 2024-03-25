#include "./linear.h"

Tensor *Linear(
  Tensor *_input,
  int64 _in,
  int64 _out
) {
  Tensor *w = xavier_w(_out, _in);
  Tensor *wx = dotprod_t(w, _input);
  Tensor *b = zero_b(wx);
  Tensor *y = add_t(wx, b);
  Tensor *ya = sigmoid_a(y);
  return ya;
}