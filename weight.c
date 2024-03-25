#include "./weight.h"

Tensor *zeros_w(
  int64 _in,
  int64 _out
) {
  size_t _ndim = 2;
  int64 _shape[] = {_in, _out};
  Tensor *w = zeros(_ndim, _shape);
  return w;
}

Tensor *ones_w(
  int64 _in,
  int64 _out
) {
  size_t _ndim = 2;
  int64 _shape[] = {_in, _out};
  Tensor *w = ones(_ndim, _shape);
  return w;
}

Tensor *const_w(
  int64 _in,
  int64 _out,
  float64 c
) {
  size_t _ndim = 2;
  int64 _shape[] = {_in, _out};
  Tensor *w = initialize(_ndim, _shape);
  for (int32 i=0; i<get_size(_ndim, _shape); i++) {
    w->value[i] = c;
  }
  return w;
}

Tensor *randn_w(
  int64 _in,
  int64 _out
) {
  size_t _ndim = 2;
  int64 _shape[] = {_in, _out};
  Tensor *w = randn(_ndim, _shape);
  return w;
}

Tensor *kaiming_w(
  int64 _in,
  int64 _out
) {
  size_t _ndim = 2;
  int64 _shape[] = {_in, _out};
  float64 mu = 0.0;
  float64 std = sqrt(2.0 / _in);
  Tensor *w = initialize(_ndim, _shape);
  for (int32 i=0; i<get_size(_ndim, _shape); i++) {
    w->value[i] = gaussian_sampling(mu, std);
  }
  return w;
}

Tensor *lecun_w(
  int64 _in,
  int64 _out
) {
  size_t _ndim = 2;
  int64 _shape[] = {_in, _out};
  float64 var = (1.0 / _in);
  float64 mu = 0.0;
  Tensor *w = initialize(_ndim, _shape);
  for (int32 i=0; i<get_size(_ndim, _shape); i++) {
    w->value[i] = gaussian_sampling(mu, var);
  }
  return w;
}

Tensor *xavier_w(
  int64 _in,
  int64 _out
) {
  size_t _ndim = 2;
  int64 _shape[] = {_in, _out};
  float64 mu = 0.0;
  float64 var = (2.0 / (_in + _out));
  Tensor *w = initialize(_ndim, _shape);
  for (int32 i=0; i<get_size(_ndim, _shape); i++) {
    w->value[i] = gaussian_sampling(mu, var);
  }
  return w;
}