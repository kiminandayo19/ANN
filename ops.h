#include "./tensor.h"

#ifndef OPS_H
#define OPS_H

/* Tensor Ops */
Tensor *add_t(
  Tensor *a,
  Tensor *b
);
Tensor *subs_t(
  Tensor *a,
  Tensor *b
);
Tensor *smul_t(
  float64 c,
  Tensor *a
);
Tensor *elmul_t(
  Tensor *a,
  Tensor *b
);
Tensor *dotprod_t(
  Tensor *a,
  Tensor *b
);

/* Common Ops */

/* Math Ops */

#endif