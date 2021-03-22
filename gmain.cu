#include <stdio.h>
#include <math.h>


__device__ void somethingElse() {
    printf("Hello, this is the gpu\n");
}


__global__ void add(int n, float* x, float* y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // somethingElse();
    for(int i = index; i < n; i += stride) y[i] = x[i] + y[i];
}


int main() {
    int N = 1<<20;
    float *x, *y, *host_x, *host_y;
    int blockSize = 512;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    host_x = (float*)malloc(sizeof(float) * N);
    host_y = (float*)malloc(sizeof(float) * N);

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        host_x[i] = 1.0f;
        host_y[i] = 2.0f;
    }

    memcpy(x, host_x, sizeof(float) * N);
    memcpy(y, host_y, sizeof(float) * N);

    // Run kernel on 1M elements on the GPU
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));

    printf("Max error: %f\n", maxError);

    // Free memory
    cudaFree(x);
    cudaFree(y);
}