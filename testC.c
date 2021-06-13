#include<stdio.h>
#include<math.h>
#include<time.h>

#include <stdlib.h>

#include "cblas.h"
//#include <complex>
//#define lapack_complex_float std::complex<float>
//#define lapack_complex_double std::complex<double>
#include "lapacke.h"

# define M_PI 3.14159265358979323846
#ifdef __cplusplus 
#undef __cplusplus 
#endif

double fibRecur(int nn) {
	if (nn <= 2)
		return(1.0);
	else
		return(fibRecur(nn - 1) + fibRecur(nn - 2));
}

double likelihood(double o, double a, double b, double h, double* y2, int N) {
	double lik = 0;
	for (int j = 1; j < N; j++) {
		h = o + a * y2[j - 1] + b * h;
		lik += log(h) + y2[j] / h;
	}
	return(lik);
}

double vvar(double* v, size_t n)
{
	double mean = 0.0;
	for (int i = 0; i < n; i++) {
		mean += v[i];
	}
	mean /= n;

	double ssd = 0.0;
	for (int i = 0; i < n; i++) {
		ssd += (v[i] - mean) * (v[i] - mean);
	}
	return ssd / (n - 1);
}

int main()
{
	int run, n;
	FILE* fp;
	fp = fopen("output.txt", "a");
	for (int run = 0; run < 5; run++)
	{
		fprintf(fp, "Run: %d\n", run + 1);
		for (n = 1000; n <= 4000; n += 200)
		{
			fprintf(fp, "Data size: %d\n", n);
			{
				//==============================
				//Matrix computations using a single-block memory allocation
				//==============================
				fprintf(fp, "Matrix computations using a single-block memory allocation.\n");
				int n2 = n * n;
				double* A = (double*)calloc(n2, sizeof(double));

				// 1: Linear Congruential Generator
				// https://en.wikipedia.org/wiki/Linear_congruential_generator

				double  a = 1331.0;
				double c = 2.718281828459045;
				double m = 34564564.0;
				A[0] = 3.141592653589793;


				// direct reference to an element of an array
				clock_t start = clock();

				for (int ii = 1; ii < n2; ii++)
				{
					A[ii] = fmod((a * A[ii - 1] + c), m);
				}

				clock_t end = clock();
				double seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixGeneration1 = %f s\n", seconds);


				// element of an array used in the next computation copied to a variable
				start = clock();

				double last;
				for (int ii = 1; ii < n2; ii++)
				{
					last = A[ii - 1];
					A[ii] = fmod((a * last + c), m);
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixGeneration2 = %f s\n", seconds);


				// 2: Matrices
				double* x = (double*)calloc(n, sizeof(double));
				for (int i = 0; i < n; i++)
				{
					x[i] = i + 1;
				}

				double* b = (double*)calloc(n, sizeof(double));

				// matrix times vector
				start = clock();

				for (int i = 0; i < n; ++i)
				{
					for (int j = 0; j < 1; ++j)
					{
						for (int k = 0; k < n; ++k)
						{
							b[i] += A[i + n * k] * x[k];
						}
					}
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixTimesVectorTime = %f s\n", seconds);


				// system of linear equations
				start = clock();
				double temp2 = 0;
				for (int i = 0; i < n; i++)
				{
					if (A[i + n * i] == 0)
					{
						for (int j = 0; j < n; j++)
						{
							if (j == i)
								continue;
							if (A[j + n * i] != 0 && A[i + n * j] != 0)
							{
								for (int k = 0; k < n; k++)
								{
									temp2 = A[j + n * k];
									A[j + n * k] = A[i + n * k];
									A[i + n * k] = temp2;
								}
								temp2 = b[j];
								b[j] = b[i];
								b[i] = temp2;
								break;
							}
						}
					}
				}

				for (int i = 0; i < n; i++)
				{
					if (A[i + n * i] == 0)
					{
						return 123;
					}
					for (int j = i + 1; j < n; j++)
					{
						double ratio = A[j + n * i] / A[i + n * i];
						for (int k = i; k < n; k++)
						{
							A[j + n * k] -= ratio * A[i + n * k];
						}
						b[j] -= ratio * b[i];
					}
				}

				x[n - 1] = b[n - 1] / A[n - 1 + n * (n - 1)];

				for (int i = n - 2; i >= 0; i--)
				{
					x[i] = b[i];
					for (int j = i + 1; j < n; j++)
					{
						x[i] = x[i] - A[i + n * j] * x[j];
					}
					x[i] = x[i] / A[i + n * i];
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "sysOfLinEqSolutionTime = %f s\n", seconds);

				free(x);
				free(b);


				double* Asqr = (double*)calloc(n2, sizeof(double));

				// matrix squared
				start = clock();

				for (int i = 0; i < n; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						for (int k = 0; k < n; ++k)
						{
							Asqr[i + n * j] += A[i + n * k] * A[k + n * j];
						}
					}
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixSquareTime = %f s\n", seconds);

				free(A);
				free(Asqr);


				double* B = (double*)calloc(n2, sizeof(double));
				double* C = (double*)calloc(n2, sizeof(double));
				double* D = (double*)calloc(n2, sizeof(double));

				// multiplication of random matrices generated using build-in tools
				start = clock();

				srand(time(NULL));
				double maxr = (double)RAND_MAX;
				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < n; j++)
					{
						B[i + n * j] = rand() / maxr;
						C[i + n * j] = rand() / maxr;
					}
				}

				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < n; j++)
					{
						for (int k = 0; k < n; k++)
						{
							D[i + n * j] += B[i + n * k] * C[k + n * j];
						}
					}
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixRandMulTime = %f s\n", seconds);

				free(B);
				free(C);
				free(D);


				double* E = (double*)calloc(n2 * 3, sizeof(double));

				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						for (int k = 0; k < 3; k++)
							E[i + n * (j + n * k)] = rand() / maxr;


				// copying parts of 3D matrix
				start = clock();

				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < n; j++)
					{
						E[i + n * (j + n * 0)] = E[i + n * (j + n * 1)];
						E[i + n * (j + n * 2)] = E[i + n * (j + n * 0)];
						E[i + n * (j + n * 1)] = E[i + n * (j + n * 2)];
					}
				}
				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrix3DCopyLoop = %f s\n", seconds);

				free(E);
			}

			{
				//==============================
				//Matrix computations using a mixed memory allocation
				//==============================
				fprintf(fp, "Matrix computations using a mixed memory allocation.\n");
				int n2 = n * n;
				double(*A)[n] = calloc(n, sizeof * A);
				double* Af = (double*)calloc(n2, sizeof(double));

				// 1: Linear Congruential Generator
				// https://en.wikipedia.org/wiki/Linear_congruential_generator
				double  a = 1331.0;
				double c = 2.718281828459045;
				double m = 34564564.0;
				Af[0] = 3.141592653589793;


				// direct reference to an element of an array
				clock_t start = clock();

				for (int ii = 1; ii < n2; ii++)
				{
					Af[ii] = fmod((a * Af[ii - 1] + c), m);
				}

				clock_t end = clock();
				double seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixGeneration1 = %f s\n", seconds);


				// element of an array used in the next computation copied to a variable
				start = clock();

				double last;
				for (int ii = 1; ii < n2; ii++)
				{
					last = Af[ii - 1];
					Af[ii] = fmod((a * last + c), m);
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixGeneration2 = %f s\n", seconds);


				// 2: Matrices
				int temp = 0;
				for (int i = 0; i < n; i++)	//reshape
				{
					for (int j = 0; j < n; j++)
					{
						A[j][i] = Af[temp];
						temp++;
					}
				}

				double* x = (double*)calloc(n, sizeof(double));
				for (int i = 0; i < n; i++)
				{
					x[i] = i + 1;
				}

				double* b = (double*)calloc(n, sizeof(double));

				// matrix times vector
				start = clock();

				for (int i = 0; i < n; ++i)
				{
					for (int j = 0; j < 1; ++j)
					{
						for (int k = 0; k < n; ++k)
						{
							b[i] += A[i][k] * x[k];
						}
					}
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixTimesVectorTime = %f s\n", seconds);
				free(Af);


				// system of linear equations
				start = clock();
				double temp2 = 0;
				for (int i = 0; i < n; i++)
				{
					if (A[i][i] == 0)
					{
						for (int j = 0; j < n; j++)
						{
							if (j == i)
								continue;
							if (A[j][i] != 0 && A[i][j] != 0)
							{
								for (int k = 0; k < n; k++)
								{
									temp2 = A[j][k];
									A[j][k] = A[i][k];
									A[i][k] = temp2;
								}
								temp2 = b[j];
								b[j] = b[i];
								b[i] = temp2;
								break;
							}
						}
					}
				}

				for (int i = 0; i < n; i++)
				{
					if (A[i][i] == 0)
					{
						return 123;
					}
					for (int j = i + 1; j < n; j++)
					{
						double ratio = A[j][i] / A[i][i];
						for (int k = i; k < n; k++)
						{
							A[j][k] -= ratio * A[i][k];
						}
						b[j] -= ratio * b[i];
					}
				}

				x[n - 1] = b[n - 1] / A[n - 1][n - 1];

				for (int i = n - 2; i >= 0; i--)
				{
					x[i] = b[i];
					for (int j = i + 1; j < n; j++)
					{
						x[i] = x[i] - A[i][j] * x[j];
					}
					x[i] = x[i] / A[i][i];
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "sysOfLinEqSolutionTime = %f s\n", seconds);

				free(x);
				free(b);


				double(*Asqr)[n] = calloc(n, sizeof * Asqr);

				// matrix squared
				start = clock();

				for (int i = 0; i < n; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						for (int k = 0; k < n; ++k)
						{
							Asqr[i][j] += A[i][k] * A[k][j];
						}
					}
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixSquareTime = %f s\n", seconds);

				free(A);
				free(Asqr);


				double(*B)[n] = calloc(n, sizeof * B);
				double(*C)[n] = calloc(n, sizeof * C);
				double(*D)[n] = calloc(n, sizeof * D);

				// multiplication of random matrices generated using build-in tools
				start = clock();

				srand(time(NULL));
				double maxr = (double)RAND_MAX;
				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < n; j++)
					{
						B[i][j] = rand() / maxr;
						C[i][j] = rand() / maxr;
					}
				}

				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < n; j++)
					{
						for (int k = 0; k < n; k++)
						{
							D[i][j] += B[i][k] * C[k][j];
						}
					}
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixRandMulTime = %f s\n", seconds);

				free(B);
				free(C);
				free(D);


				int(*E)[n][3] = calloc(n, sizeof * E);

				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						for (int k = 0; k < 3; k++)
							E[i][j][k] = rand() / maxr;


				// copying parts of 3D matrix
				start = clock();

				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < n; j++)
					{
						E[i][j][0] = E[i][j][1];
						E[i][j][2] = E[i][j][0];
						E[i][j][1] = E[i][j][2];
					}
				}
				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrix3DCopyLoop = %f s\n", seconds);

				free(E);
			}

			{
				//==============================
				//Matrix computations using a pointer memory allocation
				//==============================
				fprintf(fp, "Matrix computations using a pointer memory allocation.\n");
				int n2 = n * n;
				double** A = (double**)calloc(n, sizeof(double*));
				for (int i = 0; i < n; i++)
					A[i] = (double*)calloc(n, sizeof(double));
				double* Af = (double*)calloc(n2, sizeof(double));

				// 1: Linear Congruential Generator
				// https://en.wikipedia.org/wiki/Linear_congruential_generator
				double  a = 1331.0;
				double c = 2.718281828459045;
				double m = 34564564.0;
				Af[0] = 3.141592653589793;

				// direct reference to an element of an array
				clock_t start = clock();

				for (int ii = 1; ii < n2; ii++)
				{
					Af[ii] = fmod((a * Af[ii - 1] + c), m);
				}

				clock_t end = clock();
				double seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixGeneration1 = %f s\n", seconds);


				// element of an array used in the next computation copied to a variable
				start = clock();

				double last;
				for (int ii = 1; ii < n2; ii++)
				{
					last = Af[ii - 1];
					Af[ii] = fmod((a * last + c), m);
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixGeneration2 = %f s\n", seconds);


				// 2: Matrices
				int temp = 0;
				for (int i = 0; i < n; i++)	//reshape
				{
					for (int j = 0; j < n; j++)
					{
						A[j][i] = Af[temp];
						temp++;
					}
				}

				double* x = (double*)calloc(n, sizeof(double));
				for (int i = 0; i < n; i++)
				{
					x[i] = i + 1;
				}

				double* b = (double*)calloc(n, sizeof(double));

				// matrix times vector
				start = clock();

				for (int i = 0; i < n; ++i)
				{
					for (int j = 0; j < 1; ++j)
					{
						for (int k = 0; k < n; ++k)
						{
							b[i] += A[i][k] * x[k];
						}
					}
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixTimesVectorTime = %f s\n", seconds);
				free(Af);


				// system of linear equations
				start = clock();
				double temp2 = 0;
				for (int i = 0; i < n; i++)
				{
					if (A[i][i] == 0)
					{
						for (int j = 0; j < n; j++)
						{
							if (j == i)
								continue;
							if (A[j][i] != 0 && A[i][j] != 0)
							{
								for (int k = 0; k < n; k++)
								{
									temp2 = A[j][k];
									A[j][k] = A[i][k];
									A[i][k] = temp2;
								}
								temp2 = b[j];
								b[j] = b[i];
								b[i] = temp2;
								break;
							}
						}
					}
				}

				for (int i = 0; i < n; i++)
				{
					if (A[i][i] == 0)
					{
						return 123;
					}
					for (int j = i + 1; j < n; j++)
					{
						double ratio = A[j][i] / A[i][i];
						for (int k = i; k < n; k++)
						{
							A[j][k] -= ratio * A[i][k];
						}
						b[j] -= ratio * b[i];
					}
				}

				x[n - 1] = b[n - 1] / A[n - 1][n - 1];

				for (int i = n - 2; i >= 0; i--)
				{
					x[i] = b[i];
					for (int j = i + 1; j < n; j++)
					{
						x[i] = x[i] - A[i][j] * x[j];
					}
					x[i] = x[i] / A[i][i];
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "sysOfLinEqSolutionTime = %f s\n", seconds);

				free(x);
				free(b);


				double** Asqr = (double**)calloc(n, sizeof(double*));
				for (int i = 0; i < n; i++)
					Asqr[i] = (double*)calloc(n, sizeof(double));

				// matrix squared
				start = clock();

				for (int i = 0; i < n; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						for (int k = 0; k < n; ++k)
						{
							Asqr[i][j] += A[i][k] * A[k][j];
						}
					}
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixSquareTime = %f s\n", seconds);

				for (int i = 0; i < n; i++)
					free(A[i]);
				free(A);
				for (int i = 0; i < n; i++)
					free(Asqr[i]);
				free(Asqr);


				double** B = (double**)calloc(n, sizeof(double));
				for (int i = 0; i < n; i++)
					B[i] = (double*)calloc(n, sizeof(double));
				double** C = (double**)calloc(n, sizeof(double));
				for (int i = 0; i < n; i++)
					C[i] = (double*)calloc(n, sizeof(double));
				double** D = (double**)calloc(n, sizeof(double));
				for (int i = 0; i < n; i++)
					D[i] = (double*)calloc(n, sizeof(double));

				// multiplication of random matrices generated using build-in tools
				start = clock();

				srand(time(NULL));
				double maxr = (double)RAND_MAX;
				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < n; j++)
					{
						B[i][j] = rand() / maxr;
						C[i][j] = rand() / maxr;
					}
				}

				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < n; j++)
					{
						for (int k = 0; k < n; k++)
						{
							D[i][j] += B[i][k] * C[k][j];
						}
					}
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixRandMulTime = %f s\n", seconds);

				for (int i = 0; i < n; i++)
					free(B[i]);
				free(B);
				for (int i = 0; i < n; i++)
					free(C[i]);
				free(C);
				for (int i = 0; i < n; i++)
					free(D[i]);
				free(D);

				double*** E = (double***)calloc(n, sizeof(double**));
				for (int i = 0; i < n; i++) {
					E[i] = (double**)calloc(n, sizeof(double*));
					for (int j = 0; j < n; j++) {
						E[i][j] = (double*)calloc(3, sizeof(double));
					}
				}

				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						for (int k = 0; k < 3; k++)
							E[i][j][k] = rand() / maxr;


				// copying parts of 3D matrix
				start = clock();

				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < n; j++)
					{
						E[i][j][0] = E[i][j][1];
						E[i][j][2] = E[i][j][0];
						E[i][j][1] = E[i][j][2];
					}
				}
				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrix3DCopyLoop = %f s\n", seconds);

				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < n; j++)
					{
						free(E[i][j]);
					}
					free(E[i]);
				}
				free(E);
			}

			{
				//==============================
				//Matrix computations using a single-block memory allocation and FORTRAN routines
				//==============================
				fprintf(fp, "Matrix computations using a single-block memory allocation and FORTRAN routines.\n");
				double* Af = (double*)calloc((n * n), sizeof(double));

				// 1: Linear Congruential Generator
				// https://en.wikipedia.org/wiki/Linear_congruential_generator
				double  a = 1331.0;
				double c = 2.718281828459045;
				double m = 34564564.0;
				Af[0] = 3.141592653589793;


				// direct reference to an element of an array
				clock_t start = clock();

				for (int ii = 1; ii < (n * n); ii++)
				{
					Af[ii] = fmod((a * Af[ii - 1] + c), m);
				}

				clock_t end = clock();
				double seconds = (double)(end - start) / CLOCKS_PER_SEC;
				//fprintf(fp, "matrixGeneration1 = %f\n s", seconds);


				// element of an array used in the next computation copied to a variable
				start = clock();

				double last;
				for (int ii = 1; ii < (n * n); ii++)
				{
					last = Af[ii - 1];
					Af[ii] = fmod((a * last + c), m);
				}

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				//fprintf(fp, "matrixGeneration2 = %f\n s", seconds);


				// 2: Matrices
				double* x = (double*)calloc(n, sizeof(double));
				for (int i = 0; i < n; i++)
				{
					x[i] = i + 1.0;
				}

				double* b = (double*)calloc(n, sizeof(double));

				// matrix times vector
				start = clock();

				cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1, Af, n, x, 1, 0, b, 1);

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixTimesVectorTime = %f s\n", seconds);
				free(x);


				// system of linear equations
				start = clock();

				int* ipiv = (int*)calloc(n, sizeof(int));
				LAPACKE_dgesv(LAPACK_COL_MAJOR, n, 1, Af, n, ipiv, b, n);

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "sysOfLinEqSolutionTime = %f s\n", seconds);
				free(b);
				free(ipiv);


				double* Asqr = (double*)calloc((n * n), sizeof(double));

				// matrix squared
				start = clock();

				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 0, Af, n, Af, n, 0, Asqr, n);

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixSquareTime = %f s\n", seconds);
				free(Af);
				free(Asqr);


				double* B = (double*)calloc((n * n), sizeof(double));
				double* C = (double*)calloc((n * n), sizeof(double));
				double* D = (double*)calloc((n * n), sizeof(double));

				// multiplication of random matrices generated using build-in tools
				start = clock();

				srand(time(NULL));
				double maxr = (double)RAND_MAX;

				for (int i = 0; i < (n * n); i++)
				{
					B[i] = rand() / maxr;
					C[i] = rand() / maxr;
				}

				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 0, B, n, C, n, 0, D, n);

				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "matrixRandMulTime = %f s\n", seconds);
				free(B);
				free(C);
				free(D);

				double* E = (double*)calloc((n * n) * 3, sizeof(double));

				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						for (int k = 0; k < 3; k++)
							E[i + n * (j + n * k)] = rand() / maxr;


				// copying parts of 3D matrix
				start = clock();

				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < n; j++)
					{
						E[i + n * (j + n * 0)] = E[i + n * (j + n * 1)];
						E[i + n * (j + n * 2)] = E[i + n * (j + n * 0)];
						E[i + n * (j + n * 1)] = E[i + n * (j + n * 2)];
					}
				}
				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				//printf("matrix3DCopyLoop = %f s\n", seconds);

				free(E);
			}

			{
				//==============================
				//Numerical computations
				//==============================
				fprintf(fp, "Numerical computations.\n");
				int n2 = n * n;

				// 1: Trigonometric functions
				double* x = (double*)calloc(n2, sizeof(double));
				double val, ret, temp = -57.0;

				val = M_PI / 180;

				for (int i = 0; i < n2; i++)
				{
					x[i] = temp;
					temp++;
					if (temp == 58.0)
						temp = -57.0;
				}

				clock_t start = clock();

				for (int i = 0; i < n2; i++)
				{
					ret = sin(x[i] * val);
					ret += asin(x[i] * val);
					ret += cos(x[i] * val);
					ret += acos(x[i] * val);
					ret += tan(x[i] * val);
					ret += atan(x[i] * val);
				}

				clock_t end = clock();
				double seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "numTrig = %f s\n", seconds);
				printf("%f \n", ret);

				free(x);


				// 2: Fibonacci number - iterative
				int i = n/100;
				start = clock();
				double fPrev = 0.0, f = 1.0, fTemp;
				while (--i > 0)
				{
					fTemp = fPrev + f;
					fPrev = f;
					f = fTemp;
				}
				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "fibIterative = %f s\n", seconds);
				printf("%f \n", f);


				// 3: Fibonacci number - recursive
				start = clock();
				f = fibRecur(n/100);
				end = clock();
				seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "fibRecursive = %f s\n", seconds);
				printf("%f \n", f);
			}
		}
		for (n = 1000; n <= 10000; n += 600)
		{
			fprintf(fp, "Data size: %d\n", n);
			{
				//==============================
				//GARCH log-likelihood
				//==============================
				fprintf(fp, "GARCH log-likelihood.\n");
				FILE* fp2;
				fp2 = fopen("data.dat", "r");
				int i;
				double var = 0;
				double lik;
				double mean = 0;

				double yy;
				double* y = (double*)calloc(n, sizeof(double));
				double* y2 = (double*)calloc(n, sizeof(double));
				for (i = 0; i < n; i++) {
					fscanf(fp2, "%lf", &yy);
					y[i] = yy;
					y2[i] = yy * yy;
				}
				fclose(fp2);
				clock_t start = clock();
				for (int j = 0; j < 100; j++)
				{
					double v = vvar(y, n);
					lik = likelihood(0.001, 0.85, 0.01, v, y2, n);
				}
				clock_t end = clock();
				double seconds = (double)(end - start) / CLOCKS_PER_SEC;
				fprintf(fp, "numGARCH = %f s\n", seconds);
				printf("%f \n", lik);
				free(y);
				free(y2);
			}
		}
		fprintf(fp, "\n \n");
	}
	fclose(fp);
}
