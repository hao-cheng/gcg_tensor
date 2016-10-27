#include "math.h"
#include "mex.h"


#define MAX(a, b) (a > b ? a:b)
#define MIN(a, b) (a > b ? b:a)


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) 
{
	// Check for proper number of input and output arguments
	//nargineqchk(nrhs, 2);
	//nargouteqchk(nlhs, 2);
	// Check for input arguments type
	//argvectorchk(prhs, 0);
	//argvectorchk(prhs, 1);

	double* v = mxGetPr(prhs[0]);
	double* sz = mxGetPr(prhs[1]);
  double* res, *normVec;
	int sz_len = MAX(mxGetM(prhs[1]), mxGetN(prhs[1]));
	if(MIN(mxGetM(prhs[1]), mxGetN(prhs[1]) == 0))
	{
		mexErrMsgTxt("sz can not be empty!");
	}

	int v_len = MAX(mxGetM(prhs[0]), mxGetN(prhs[0]));
  int dim1 = (int)(sz[0]);
  int dim2 = (int)(sz[1]);
  int dim3 = (int)(sz[2]);
  int nnz = dim1*dim2*dim3;
  int number_of_dims = 3;
  mwSize dims[3];
  dims[0] = dim1; dims[1] = dim2; dims[2] = dim3;

  int sum_sz = dim1+dim2+dim3;
  int rank = v_len / sum_sz;
  int U_len = dim1 * dim2 * dim3;  

  plhs[0] = mxCreateNumericArray(number_of_dims, (const mwSize*)dims, mxDOUBLE_CLASS, mxREAL);
  res = mxGetPr(plhs[0]);
//   normVec = mxGetPr(plhs[1]);
  double *i_start = v;
  double *j_start = v + dim1;
  double *k_start = v + dim1 + dim2;
  double tmp;
  for (int r = 0; r < rank; r ++)
  {    
    for (double *k = k_start; k < k_start + dim3; k ++)
    {      
      for (double *j = j_start; j < j_start + dim2; j ++)
      {
        tmp = *k * *j;
        for (double *i = i_start; i < i_start + dim1; i ++)
        {
          *(res++) += *i * tmp;
        }
      }
    }
    i_start += sum_sz;  j_start += sum_sz;  k_start += sum_sz;
    res -= nnz;
  }
}