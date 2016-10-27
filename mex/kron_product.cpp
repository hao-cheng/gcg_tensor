#include "mex.h"
#include <math.h>

// To be used as
//  [R] = kron_product(T, u)
// Input: 
//   T: a tensor (can be represented as a vector)
//   u: a vector
// Output:
//   R: the result of T \kron u represented as a vector
void mexFunction (int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray *prhs[]) 
{
  int num_ele_tensor = mxGetNumberOfElements(prhs[0]);
  int dim1 = mxGetNumberOfElements(prhs[1]);
  int len_res = num_ele_tensor * dim1;

  plhs[0] = mxCreateDoubleMatrix(len_res, 1, mxREAL);

  double *ptr_u, *ptr_T, *ptr3, tmp;
  ptr_T = mxGetPr(prhs[0]);
  ptr_u = mxGetPr(prhs[1]);
  ptr3 = mxGetPr(plhs[0]);
  for (int i = 0; i < num_ele_tensor; i ++)
  {
    tmp = *(ptr_T++);
    for (int j = 0; j < dim1; j ++)
      *(ptr3 ++) = tmp * *(ptr_u++);

    ptr_u -= dim1;
  }
}
