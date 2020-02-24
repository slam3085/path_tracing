#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "ray_pathing.h"
#include "ray.h"

__device__ bool hit_sphere(Ray* r)
{
    vec3 center = { 0, 0, -1 };
    float radius = 0.5;
    vec3 oc = r->origin - center;
    float a = dot(r->direction, r->direction);
    float b = 2.0 * dot(oc, r->direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    return discriminant > 0;
}

__device__ vec3 color(Ray* ray)
{
    if (hit_sphere(ray))
        return { 1, 0, 0 };
    vec3 unit_direction = ray->direction.unit_vector();
    float t = 0.5 * unit_direction.Y + 1.0;
    return {
        1.0 - t + 0.5 * t,
        1.0 - t + 0.7 * t,
        1.0 - t + 1.0 * t
    };
}

__global__ void ray_pathing_kernel(vec3* dev_framebuffer, int height, int width)
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

cudaError_t ray_pathing_with_cuda(vec3* framebuffer, int height, int width)
{
    int size = width * height;
    cudaError_t cudaStatus = cudaSetDevice(0);
    vec3* dev_framebuffer = 0;
    cudaStatus = cudaMalloc((void**)&dev_framebuffer, size * sizeof(vec3));
    cudaStatus = cudaMemcpy(dev_framebuffer, framebuffer, size * sizeof(vec3), cudaMemcpyHostToDevice);
    ray_pathing_kernel <<<height, width >>>(dev_framebuffer, height, width);
    cudaStatus = cudaGetLastError();
    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(framebuffer, dev_framebuffer, size * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaFree(dev_framebuffer);
    return cudaStatus;
}
