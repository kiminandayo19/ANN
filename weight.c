#include "./weight.h"

Tensor weight_init(
  int64 in,
  int64 out,
  Method method,
  float64 k
) {
  size_t ndim = 2;
  int64 shape[] = {in, out};
  Tensor w = initialize(ndim, shape);

  switch (method) {
    case ZEROS:
      w = zeros_w(in, out);
      break;
    case ONES:
      w = ones_w(in, out);
      break;
    case CONSTANT:
      w = const_w(in, out, k);
      break;
    case RANDN:
      w = randn_w(in, out);
      break;
    case KAIMING_HE:
      w = kaiming_he(in, out);
      break;
    case LECUN:
      w = lecun(in, out);
      break;
    case XAVIER:
      w = xavier(in, out);
      break;
    case ID:
      w = id_w(in, out);
      break;
  }
  return w;
}

Tensor zeros_w(
  int64 in_feat,
  int64 out_feat
) {
  size_t ndim = 2;
  int64 shape[] = {in_feat, out_feat};
  return zeros(ndim, shape);
}

Tensor ones_w(
  int64 in_feat,
  int64 out_feat
) {
  size_t ndim = 2;
  int64 shape[] = {in_feat, out_feat};
  return ones(ndim, shape);
}

Tensor const_w(
  int64 in_feat,
  int64 out_feat,
  float64 k
) {
  size_t ndim = 2;
  int64 shape[] = {in_feat, out_feat};
  Tensor cw = initialize(ndim, shape);
  assign_scalar(&cw, k);
  return cw;
}

Tensor randn_w(
  int64 in_feat,
  int64 out_feat
) {
  size_t ndim = 2;
  int64 shape[] = {in_feat, out_feat};
  return randn(ndim, shape);
}

Tensor kaiming_he(
  int64 in_feat,
  int64 out_feat
) {
  size_t ndim = 2;
  int64 shape[] = {in_feat, out_feat};
  float64 stdev = sqrt((2.0/in_feat));
  return cust_randn(ndim, shape, 0.0, stdev);
}

Tensor lecun(
  int64 in_feat,
  int64 out_feat
) {
  size_t ndim = 2;
  int64 shape[] = {in_feat, out_feat};
  return zeros(ndim, shape);
}

Tensor xavier(
  int64 in_feat,
  int64 out_feat
) {
  size_t ndim = 2;
  int64 shape[] = {in_feat, out_feat};
  return zeros(ndim, shape);
}

Tensor id_w(
  int64 in_feat,
  int64 out_feat
) {
  size_t ndim = 2;
  int64 shape[] = {in_feat, out_feat};
  int64 size = get_size(ndim, shape);
  Tensor idw = initialize(ndim, shape);
  for (int32 i=0; i<size; i++)
    idw.value[i * in_feat + i] = 1.0;
  return idw;
}

Tensor bias_init(int64 out_feat)
{
  size_t ndim = 1;
  int64 shape[] = {out_feat};
  return zeros(ndim, shape);
}

Tensor forward(
  Tensor *W,
  Tensor *X,
  Tensor *b
) {
  Tensor mul = matrix_vector_mul(W, X);
  return add(&mul, b);
}

Tensor activation(
  Tensor *Y,
  Activation activation
);

Tensor Linear(
  int64 in_features,
  int64 out_features,
  Activation activation
);

int main() {
  // !TODO: Implement Activation Function
  // !TODO: Create Test Case
  // !TODO: Create Forward Propagation
  return 0;
}