#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "path_tracing.h"
#include "ray.h"
#include "sphere.h"
#include "hittable.h"
#include "random.h"
#include "camera.h"
#include "material.h"

__global__ void init_common(Hittable** dev_world, Camera** dev_camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Hittable** list = new Hittable*[5];
        list[0] = new Sphere({ 0, 0, -1 }, 0.49, new Lambertian({ 0.1, 0.2, 0.5 }));
        list[1] = new Sphere({ 0, -100.5, -1 }, 99.99, new Lambertian({ 0.8, 0.8, 0.0 }));
        list[2] = new Sphere({ 1, 0, -1 }, 0.49, new Metal({ 0.8, 0.6, 0.2 }, 0.3));
        list[3] = new Sphere({ -1, 0, -1 }, 0.49, new Dielectric(1.5));
        list[4] = new Sphere({ -1, 0, -1 }, -0.45, new Dielectric(1.5));
        *dev_world = new HittableList(list, 5);
        *dev_camera = new Camera();
    }
}

__global__ void init_curand(curandState_t* states, int height, int width)
{
    int size = height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        curand_init(1234, idx, 0, &states[idx]);
}

__device__ vec3 color(Ray* ray, Hittable** dev_world, curandState_t* state)
{
    int depth = 50;
    vec3 col = { 1.0, 1.0, 1.0 };
    HitRecord rec;
    Ray scattered;
    vec3 attenuation;
    while((*dev_world)->hit(ray, 0.001, 1E38, &rec))
    {
        if (depth && rec.material->scatter(ray, &rec, &attenuation, &scattered, state))
        {
            col *= attenuation;
            depth -= 1;
            ray = &scattered;
        }
        else
            return { 0, 0, 0 };
    }
    vec3 unit_direction = ray->direction.unit_vector();
    float t = 0.5 * (unit_direction.Y + 1.0);
    col.X *= (1.0 - 0.5 * t);
    col.Y *= (1.0 - 0.3 * t);
    return col;
}

__global__ void path_tracing_kernel(Hittable** dev_world, Camera** dev_camera, vec3* dev_framebuffer, int height, int width, curandState_t* states)
{
    int size = width * height;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        int i = idx % width;
        int j = idx / width;
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
    init_curand<<<size / 256 + 1, 256>>>(states, height, width);
    //tracing
    path_tracing_kernel<<<size / 512 + 1, 512>>>(dev_world, dev_camera, dev_framebuffer, height, width, states);
    cudaDeviceSynchronize();
    cudaMemcpy(framebuffer, dev_framebuffer, size * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaFree(dev_framebuffer);
    cudaFree(dev_world);
    cudaFree(dev_camera);
}
