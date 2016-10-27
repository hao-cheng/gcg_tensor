#include "common.c"
#include "math.h"
#include "blas.h"


#define MAX(a, b) (a > b ? a:b)
#define MIN(a, b) (a > b ? b:a)


/** Kronecker product of a real A and a real B. */
void KronReal(const double* A, int ma, int na, const double* B, int mb, int nb, double* C) {

	int i,j,k,l;
	const double* pa;
	const double* pb;
	double a,b;

	for (j = 0; j < na; j++) {  
		/* iterate over columns of A */
		for (l = 0; l < nb; l++) {  
			/* iterative over columns of B */
			pa = &A[j*ma];
			for (i = 0; i < ma; i++) {  
				/* iterate over rows of A */
				a = (*pa++);
				if (!isnonzero(a)) {  
					/* no need to multiply by zero */
					C += mb;
					continue;
				}
				pb = &B[l*mb];  
				/* position to lth column of B */
				for (k = 0; k < mb; k++) {  
					/* iterate over rows of B */
					b = *(pb++);
					*(C++) = a * b;
				}
			}
		}
	}
}

double norm(const double* u, int size)
{
	double val = 0;
	for(int i = 0; i < size; i++)
	{
		val += (u[i] * u[i]);
	}
	val = sqrt(val);
	//mexPrintf("Norm val %f\n", val);
	return val;
}

void copyVal(double* temp, double* U, int start, int end)
{
	for(int i = start; i < end; i++)
	{
		temp[i] = U[i];
	}
}

double vecElementSum(const double* vec, int startIdx, int endIdx)
{
	double sum = 0.0;
	for(int i = startIdx; i < endIdx; i++)
	{
		sum += vec[i];
	}
	return sum;
}

double vecElementProd(const double* vec, int startIdx, int endIdx)
{
	double product = 1.0;
	for(int i = startIdx; i < endIdx; i++)
	{
		product *= vec[i];
	}
	return product;
}

void makeCopyConstDouble(const double* from, double* to, int len)
{
	for(int i = 0; i < len; i++)
	{
		to[i] = from[i];
	}

}

void matElementAdd(double* A, double* B, int size)
{
	/*
	 * Add mat A and B, store the result in A
	 */
	for(int i = 0; i < size; i++)
	{
		A[i] += B[i];
	}

}


/*
 * Rank-one tensor generation
 *	[U, n(ii)] = RankOneTensorGenerate(u, sz)
 */
void modeOneUnfolding(double* t, double* normVec, int idx, const double* u, const double* sz, int U_len, int sz_len)
{
	double* uFirst = mxGetPr(mxCreateDoubleMatrix((int)sz[0], 1, mxREAL));
	double* U = mxGetPr(mxCreateDoubleMatrix(1, U_len, mxREAL));
	double* t_cur = mxGetPr(mxCreateDoubleMatrix((int)sz[0], U_len, mxREAL));
	double normVal = 1.0;

	const double* u_cur;
	double* tempU = mxGetPr(mxCreateDoubleMatrix(U_len, 1, mxREAL));

	int ind = 0;
	int curSize = 1;
	tempU[0] = 1;
	for(int k = 1; k < sz_len; k++)
	{
		ind += (int)sz[k - 1];
		int mu = sz[k];
		u_cur = &(u[ind]);
		KronReal(tempU, curSize, 1, u_cur, mu, 1, U);
		curSize *= mu;
		copyVal(tempU, U, 0, curSize);
		normVal *= norm(u_cur, mu);
	}

	// Pass arguments to Fortran by reference and compute matrix product

	makeCopyConstDouble(u, uFirst, (int)sz[0]);


	mwSignedIndex m, n, p;
	m = sz[0];
	p = 1;
	n = U_len;
	char *chn = "N";
	double one = 1.0, zero = 0.0;

	dgemm(chn, chn, &m, &n, &p, &one, uFirst, &m, U, &p, &zero, t_cur, &m);
	matElementAdd(t, t_cur, U_len * ((int) sz[0]));

	normVal *= norm(u, (int) sz[0]);
	normVec[idx] = normVal;
}




void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) 
{
	// Check for proper number of input and output arguments
	//nargineqchk(nrhs, 2);
	//nargouteqchk(nlhs, 2);
	// Check for input arguments type
	//argvectorchk(prhs, 0);
	//argvectorchk(prhs, 1);

	const double* v = mxGetPr(prhs[0]);
	const double* sz = mxGetPr(prhs[1]);
	int sz_len = MAX(mxGetM(prhs[1]), mxGetN(prhs[1]));
	if(MIN(mxGetM(prhs[1]), mxGetN(prhs[1]) == 0))
	{
		mexErrMsgTxt("sz can not be empty!");
	}

	int v_len = MAX(mxGetM(prhs[0]), mxGetN(prhs[0]));
	bool empty = MIN(mxGetM(prhs[0]), mxGetN(prhs[0])) == 0;

	int sum_sz = vecElementSum(sz, 0, sz_len);
	int rank = empty ? 1 : (v_len / sum_sz);
	//return;

	double* t;
	double* normVec;

	int U_len = (int) vecElementProd(sz, 1, sz_len);

	// Initialize t matrix with size sz[0]x(sz[1]x...sz[sz_len-1])
	plhs[0] = mxCreateDoubleMatrix((int)sz[0], U_len, mxREAL);
	t = mxGetPr(plhs[0]);
	// Initialize normVec with size rank x 1
	plhs[1] = mxCreateDoubleMatrix(rank, 1, mxREAL);
	normVec = mxGetPr(plhs[1]);
	//return;
	if(empty)
	{
		// Empty v case
		return;
	}

	for(unsigned ii = 0; ii < rank; ii++)
	{
		int idx = ii * sum_sz;
		const double* u = &v[idx];

		modeOneUnfolding(t, normVec, ii, u, sz, U_len, sz_len);

	}

}