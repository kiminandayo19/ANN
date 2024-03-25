#include "./datatype.h"
#include "./tensor.h"
#include "./activation.h"
#include "./weight.h"
#include "./ops.h"

#ifndef BIAS_H
#define BIAS_H

Tensor *zero_b(
  Tensor *y
);

#endif