#include "mex.h"
#include <math.h>

// To be used as
//  [R] = mode1_product(T, u)
// Input: 
//   T: a tensor (can be represented as a vector)
//   u: a vector
// Output:
//   R: the result of T *_1 u represented as a vector
void mexFunction (int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray *prhs[]) 
{
  int num_ele_tensor = mxGetNumberOfElements(prhs[0]);
  int dim1 = mxGetNumberOfElements(prhs[1]);
  int len_res = num_ele_tensor / dim1;

  plhs[0] = mxCreateDoubleMatrix(len_res, 1, mxREAL);

  double *ptr1, *ptr2, *ptr3, res;
  ptr1 = mxGetPr(prhs[0]);
  ptr2 = mxGetPr(prhs[1]);
  ptr3 = mxGetPr(plhs[0]);
  for (int i = 0; i < len_res; i ++)
  {
    res = 0.0;
    for (int j = 0; j < dim1; j ++)
      res += *(ptr1++) * *(ptr2++);

    *(ptr3 ++) = res;
    ptr2 -= dim1;
  }
}
