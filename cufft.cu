#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cuComplex.h>
#include <assert.h>
#include <cufft.h>
#include <cstdlib>
#include <cstring>
#define DEBUG
using namespace std;
const int N = 2e5 + 10;
int t, n;
__constant__ int T[1];

inline cudaError_t checkCuda(cudaError_t result){
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

__global__  void vector_mul(cufftDoubleComplex *a, cufftDoubleComplex *b){
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < T[0]; i += numThreads) {//why can't i use *T in here
		cuDoubleComplex c = cuCmul(a[i], b[i]);
		a[i] = make_cuDoubleComplex(cuCreal(c) / T[0], cuCimag(c) / T[0]);
	}
}

__global__ void get_ans(int *ans, cufftDoubleComplex *a) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < T[0]; i += numThreads)//why can't i use *T in here
		ans[i] = (int)(cuCreal(a[i]) + 0.5);

}

int main(){
	cudaDeviceProp prop;
	checkCuda(cudaGetDeviceProperties(&prop, 0));
	//timing start
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//intialize
	FILE *in=fopen("fft.in","r"), *out=fopen("cufft.out","w");
	fscanf(in, "%d", &n);
	t = 1; while (t < n + n) t <<= 1;
	//memory allocation
	int tt[1]; tt[0] = t;
	checkCuda(cudaMemcpyToSymbol(T, tt, sizeof(int)));
	int size = sizeof(cufftDoubleComplex)*t, size2 = sizeof(int)*t;
	int* h_ans = (int*)calloc(t, sizeof(int));
	char* str = (char*)malloc(sizeof(char)*(n + 10));
	int* ans;
	checkCuda(cudaMalloc((void **)&ans, size2));
	cufftDoubleComplex *a, *b;
	cufftDoubleComplex *h_a = (cufftDoubleComplex *)calloc(t, sizeof(cufftDoubleComplex));
	cufftDoubleComplex *h_b = (cufftDoubleComplex *)calloc(t, sizeof(cufftDoubleComplex));
	checkCuda(cudaMalloc((void **)&a, size));
	checkCuda(cudaMalloc((void **)&b, size));
	//input and memcpy
	fscanf(in, "%s", str); for (int i = 0; i < n; i++) h_a[i] = make_cuDoubleComplex((double)str[n - i - 1] - '0', 0.0);
	fscanf(in, "%s", str); for (int i = 0; i < n; i++) h_b[i] = make_cuDoubleComplex((double)str[n - i - 1] - '0', 0.0);
	checkCuda(cudaMemcpy(a, h_a, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(b, h_b, size, cudaMemcpyHostToDevice));
	//dft
	cufftHandle plan;
	if (cufftPlan1d(&plan, t, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
		fprintf(stderr, "cufft plan create failed!");
		return 1;
	}
	if (cufftExecZ2Z(plan, a, a, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: a ExecZ2Z Forward failed");
		return 2;
	}
	if (cufftExecZ2Z(plan, b, b, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: b ExecZ2Z Forward failed");
		return 2;
	}
	//multiply
	vector_mul<<<t / prop.maxThreadsPerBlock + 1, prop.maxThreadsPerBlock>>>(a, b);
	checkCuda(cudaDeviceSynchronize());
	//idft
	if (cufftExecZ2Z(plan, a, a, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: a ExecZ2Z Inverse failed");
		return 4;
	}
	if (cufftDestroy(plan) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: fft plan destroy failed");
		return 5;
	}
	//change into integer: serial or parallel??
	checkCuda(cudaMemcpy(h_a, a, size, cudaMemcpyDeviceToHost));
	//for (int i = 0; i<t; i++) ans[i] = (int)(cuCreal(h_a[i]) + 0.5);
	//for (int i = 0; i<t; i++) ans[i + 1] += ans[i] / 10, ans[i] %= 10;
	get_ans<<<t / prop.maxThreadsPerBlock + 1, prop.maxThreadsPerBlock >>>(ans, a);
	checkCuda(cudaMemcpy(h_ans, ans, size2, cudaMemcpyDeviceToHost));
	for (int i = 0; i<t; i++) h_ans[i + 1] += h_ans[i] / 10, h_ans[i] %= 10;
	//output
	while (!h_ans[t-1]) t--;
	for (int i = t-1; i >= 0; i--) fprintf(out, "%d", h_ans[i]);
	//delete
	checkCuda(cudaFree(a));
	checkCuda(cudaFree(b));
	checkCuda(cudaFree(ans));
	free(h_ans);
	free(h_a);
	free(h_b);
	free(str);
	//timing end
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%lf\t", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if (checkCuda(cudaDeviceReset()) != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 10;
	}
	return 0;
}