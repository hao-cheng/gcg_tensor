#include "mex.h"
#include <math.h>

// To be used as
//  [g] = final_product(F, B)
// Input: 
//   F: a tensor (can be represented as a vector)
//   B: a tensor made by Kronecker product of vectors
// Output:
//   g: the result of F_(k) * B
void mexFunction (int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray *prhs[]) 
{
  int num_ele_F = mxGetNumberOfElements(prhs[0]);
  int num_ele_B = mxGetNumberOfElements(prhs[1]);
  int dim1 = num_ele_F / num_ele_B;

  plhs[0] = mxCreateDoubleMatrix(dim1, 1, mxREAL);

  double *ptr_F, *ptr_B, *ptr3, res;
  ptr_F = mxGetPr(prhs[0]);
  ptr_B = mxGetPr(prhs[1]);
  ptr3 = mxGetPr(plhs[0]);
  for (int i = 0; i < dim1; i ++)
  {
    res = 0.0;
    for (int j = 0; j < num_ele_B; j ++)
    {
      res += *ptr_F * *(ptr_B++);
      ptr_F += dim1;
    }

    *(ptr3 ++) = res;
    ptr_B -= num_ele_B;
    ptr_F -= num_ele_F - 1;
  }
}
