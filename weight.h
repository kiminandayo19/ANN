#include "./tensor.h"
#include "./ops.h"

#ifndef WEIGHT_H
#define WEIGHT_H

Tensor *zeros_w(
  int64 _in,
  int64 _out
);
Tensor *ones_w(
  int64 _in,
  int64 _out
);
Tensor *const_w(
  int64 _in,
  int64 _out,
  float64 c
);
Tensor *randn_w(
  int64 _in,
  int64 _out
);
Tensor *kaiming_w(
  int64 _in,
  int64 _out
);
Tensor *lecun_w(
  int64 _in,
  int64 _out
);
Tensor *xavier_w(
  int64 _in,
  int64 _out
);

#endif