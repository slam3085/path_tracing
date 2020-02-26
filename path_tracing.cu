#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "path_tracing.h"
#include "ray.h"
#include "sphere.h"
#include "hittable.h"
#include "random.h"
#include "camera.h"

__global__ void init_common(Hittable** dev_world, Camera** dev_camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Hittable** list = new Hittable*[2];
        list[0] = new Sphere({ 0, 0, -1 }, 0.5);
        list[1] = new Sphere({ 0, -100.5, -1 }, 100);
        *dev_world = new HittableList(list, 2);
        *dev_camera = new Camera();
    }
}

__device__ vec3 color(Ray* ray, Hittable** dev_world, curandState_t* state)
{
    float multiplier = 1.0;
    HitRecord rec;
    while((*dev_world)->hit(ray, 0.001, 1E38, &rec))
    {
        vec3 target = rec.p + rec.normal + random_unit_in_sphere(state);
        ray->origin = rec.p;
        ray->direction = target - rec.p;
        multiplier *= 0.5;
    }
    vec3 unit_direction = ray->direction.unit_vector();
    float t = 0.5 * (unit_direction.Y + 1.0);
    return {
        multiplier * (1.0 - 0.5 * t),
        multiplier * (1.0 - 0.3 * t),
        multiplier
    };
}

__global__ void path_tracing_kernel(Hittable** dev_world, Camera** dev_camera, vec3* dev_framebuffer, int height, int width, curandState_t* states)
{
    int size = width * height;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        int i = idx % width;
        int j = idx / width;
        //rand init
        //curand_init(0, 0, 0, &states[idx]);
        //crash -_-
        //rays
        int ns = 1000;
        vec3 col = { 0, 0, 0 };
        for (int s = 0; s < ns; s++)
        {
            float u = (float(i) + random_float(&states[idx])) / float(width);
            float v = (float(j) + random_float(&states[idx])) / float(height);
            Ray ray = (*dev_camera)->get_ray(u, v);
            col += color(&ray, dev_world, &states[idx]);
        }
        col /= float(ns);
        col.X = sqrt(col.X);
        col.Y = sqrt(col.Y);
        col.Z = sqrt(col.Z);
        dev_framebuffer[idx] = col;
    }
}

void path_tracing_with_cuda(vec3* framebuffer, int height, int width)
{
    cudaSetDevice(0);
    //framebuffer
    int size = width * height;
    vec3* dev_framebuffer = 0;
    cudaMalloc((void**)&dev_framebuffer, size * sizeof(vec3));
    //camera
    Camera** dev_camera = 0;
    cudaMalloc(&dev_camera, sizeof(Camera**));
    //world
    Hittable** dev_world = 0;
    cudaMalloc(&dev_world, sizeof(Hittable**));
    init_common <<<1,1>>>(dev_world, dev_camera);
    //various
    curandState_t* states = 0;
    cudaMalloc((void**)&states, size * sizeof(curandState_t));
    //tracing
    path_tracing_kernel <<<size / 512 + 1, 512>>>(dev_world, dev_camera, dev_framebuffer, height, width, states);
    cudaDeviceSynchronize();
    cudaMemcpy(framebuffer, dev_framebuffer, size * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaFree(dev_framebuffer);
    cudaFree(dev_world);
    cudaFree(dev_camera);
}
