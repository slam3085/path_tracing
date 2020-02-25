#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "path_tracing.h"
#include "ray.h"
#include "sphere.h"
#include "hittable.h"
#include "stdio.h"

__global__ void init_world(Hittable** dev_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        dev_world[0] = new Sphere({ 0, 0, -1 }, 0.5);
        dev_world[1] = new Sphere({ 0, -100.5, -1 }, 100);
    }
}

__device__ vec3 color(Ray* ray, Hittable** dev_world, int world_size)
{
    HitRecord rec;
    float t_min = 0.0, t_max = 1E9;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < world_size; i++)
    {
        if (dev_world[i]->hit(ray, t_min, closest_so_far, &rec))
        {
            hit_anything = true;
            closest_so_far = rec.t;
        }
    }
    if(hit_anything)
    {
        vec3 ones = { 1, 1, 1 };
        return (rec.normal + ones) * 0.5;
    }
    vec3 unit_direction = ray->direction.unit_vector();
    float t = 0.5 * (unit_direction.Y + 1.0);
    return {
        1.0 - t + 0.5 * t,
        1.0 - t + 0.7 * t,
        1.0 - t + 1.0 * t
    };
}

__global__ void path_tracing_kernel(Hittable** dev_world, int world_size, vec3* dev_framebuffer, int height, int width)
{
    int size = width * height;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        int i = idx % width;
        int j = idx / width;
        // camera
        vec3 lower_left_corner = { -2, -1, -1 };
        vec3 horizontal = { 4, 0, 0 };
        vec3 vertical = { 0, 2, 0 };
        vec3 origin = { 0, 0, 0 };
        //ray
        float u = float(i) / float(width);
        float v = float(j) / float(height);
        Ray ray = {
            origin, 
            lower_left_corner + horizontal * u + vertical * v
        };
        dev_framebuffer[idx] = color(&ray, dev_world, world_size);
    }
}

cudaError_t path_tracing_with_cuda(vec3* framebuffer, int height, int width)
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    //framebuffer
    int size = width * height;
    vec3* dev_framebuffer = 0;
    cudaStatus = cudaMalloc((void**)&dev_framebuffer, size * sizeof(vec3));
    //world
    int world_size = 2;
    Hittable** dev_world = 0;
    cudaStatus = cudaMalloc(&dev_world, world_size * sizeof(Hittable**));
    init_world <<<1,1>>>(dev_world);
    //tracing
    path_tracing_kernel <<<size / 512 + 1, 512>>>(dev_world, world_size, dev_framebuffer, height, width);
    cudaStatus = cudaGetLastError();
    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(framebuffer, dev_framebuffer, size * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaFree(dev_framebuffer);
    cudaFree(dev_world);
    return cudaStatus;
}
