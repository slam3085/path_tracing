#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gradient.h"

__global__ void gradientKernel(float* dev_framebuffer, int height, int width)
{
	int size = 3 * width * height;
	int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	if (i < size)
	{
		dev_framebuffer[i] = float(threadIdx.x) / float(width);
		dev_framebuffer[i + 1] = float(blockIdx.x) / float(height);
		dev_framebuffer[i + 2] = 0.2;
	}
}

cudaError_t gradientWithCuda(float* framebuffer, int height, int width)
{
	int size = 3 * width * height;
	cudaError_t cudaStatus = cudaSetDevice(0);
	float* dev_framebuffer = 0;
	cudaStatus = cudaMalloc((void**)&dev_framebuffer, size * sizeof(float));
    cudaStatus = cudaMemcpy(dev_framebuffer, framebuffer, size * sizeof(float), cudaMemcpyHostToDevice);
	gradientKernel <<<height, width >>>(dev_framebuffer, height, width);
    cudaStatus = cudaGetLastError();
    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(framebuffer, dev_framebuffer, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_framebuffer);
    return cudaStatus;
}
