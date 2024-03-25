#include "./datatype.h"
#include "./tensor.h"
#include "./ops.h"
#include "./weight.h"
#include "./bias.h"
#include "./activation.h"

#ifndef LINEAR_H
#define LINEAR_H

Tensor *Linear(
  Tensor *_input,
  int64 _in,
  int64 _out
);

#endif