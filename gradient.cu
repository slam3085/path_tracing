#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "gradient.h"
#include "ray.h"

__device__ vec3 color(Ray* ray)
{
    vec3 unit_direction = ray->direction.unit_vector();
    float t = 0.5 * unit_direction.Y + 1.0;
    return {
        1.0 - t + 0.5 * t,
        1.0 - t + 0.7 * t,
        1.0 - t + 1.0 * t
    };
}

__global__ void gradientKernel(vec3* dev_framebuffer, int height, int width)
{
    int size = width * height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        vec3 lower_left_corner = { -2, -1, -1 };
        vec3 horizontal = { 4, 0, 0 };
        vec3 vertical = { 0, 2, 0 };
        vec3 origin = { 0, 0, 0 };
        float u = float(threadIdx.x) / float(width);
        float v = float(blockIdx.x) / float(height);
        Ray ray = {
            origin, 
            lower_left_corner + horizontal * u + vertical * v
        };
        dev_framebuffer[i] = color(&ray);
    }
}

cudaError_t gradientWithCuda(vec3* framebuffer, int height, int width)
{
    int size = width * height;
    cudaError_t cudaStatus = cudaSetDevice(0);
    vec3* dev_framebuffer = 0;
    cudaStatus = cudaMalloc((void**)&dev_framebuffer, size * sizeof(vec3));
    cudaStatus = cudaMemcpy(dev_framebuffer, framebuffer, size * sizeof(vec3), cudaMemcpyHostToDevice);
    gradientKernel <<<height, width >>>(dev_framebuffer, height, width);
    cudaStatus = cudaGetLastError();
    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(framebuffer, dev_framebuffer, size * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaFree(dev_framebuffer);
    return cudaStatus;
}
