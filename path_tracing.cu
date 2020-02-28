#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define _USE_MATH_DEFINES
#include "math.h"
#include "path_tracing.h"
#include "ray.h"
#include "sphere.h"
#include "hittable.h"
#include "random.h"
#include "camera.h"
#include "material.h"

__global__ void init_common(Hittable** dev_world, Camera** dev_camera, int height, int width, curandState_t* states)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        int n = 250;
        Hittable** list = new Hittable*[n + 1];
        list[0] = new Sphere({ 0, -1000, 0 }, 1000, new Lambertian({ 0.5, 0.5, 0.5 }));
        int i = 1;
        vec3 tmp = { 4, 0.2, 0 };
        for(int a = -5; a < 5; a++)
            for(int b = -5; b < 5; b++)
                {
                    vec3 center = { a + 0.9 * random_float(&states[0]), 0.2, b + 0.9 * random_float(&states[1]) };
                    if((center - tmp).length() > 0.9)
                    {
                        Material* material;
                        float mat_rnd = random_float(&states[0]);
                        if (mat_rnd < 0.8)
                            material = new Lambertian({ random_float(&states[0]) * random_float(&states[1]), random_float(&states[2]) * random_float(&states[3]), random_float(&states[4]) * random_float(&states[5]) });
                        else if (mat_rnd < 0.95)
                            material = new Metal({ 0.5f * (1.0f + random_float(&states[0])), 0.5f * (1.0f + random_float(&states[1])), 0.5f * (1.0f + random_float(&states[2])) }, 0.5 * random_float(&states[3]));
                        else
                            material = new Dielectric(1.5);
                        list[i++] = new Sphere(center, 0.2, material);
                    }
                    
                }
        list[i++] = new Sphere({ 0, 1, 0 }, 1.0, new Metal({ 0.7, 0.6, 0.5 }, 0.0));
        list[i++] = new Sphere({ -4, 1, 0 }, 1.0, new Lambertian({ 0.4, 0.2, 0.1 }));
        list[i++] = new Sphere({ 4, 1, 0 }, 1.0, new Dielectric(1.5));
        list[i++] = new Sphere({ 4, 1, 0 }, -0.95, new Dielectric(1.5));
        *dev_world = new HittableList(list, i);
        *dev_camera = new Camera({ 13, 2, 3 }, { 0, 0, 0 }, { 0, 1, 0 }, 20, float(width) / float(height));
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
    vec3 col = { 1.0f, 1.0f, 1.0f };
    HitRecord rec;
    Ray scattered;
    vec3 attenuation;
    while((*dev_world)->hit(ray, 0.001f, 1E38f, &rec))
    {
        if (depth && rec.material->scatter(ray, &rec, &attenuation, &scattered, state))
        {
            col *= attenuation;
            depth -= 1;
            ray = &scattered;
        }
        else
            return { 0.0f, 0.0f, 0.0f };
    }
    vec3 unit_direction = ray->direction.unit_vector();
    float t = 0.5f * (unit_direction.Y + 1.0f);
    col.X *= (1.0f - 0.5f * t);
    col.Y *= (1.0f - 0.3f * t);
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
        vec3 col = { 0.0f, 0.0f, 0.0f };
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
    //curand
    curandState_t* states = 0;
    cudaMalloc((void**)&states, size * sizeof(curandState_t));
    init_curand<<<size / 256 + 1, 256>>>(states, height, width);
    //world
    Hittable** dev_world = 0;
    cudaMalloc(&dev_world, sizeof(Hittable**));
    init_common <<<1,1>>>(dev_world, dev_camera, height, width, states);
    //tracing
    path_tracing_kernel<<<size / 256 + 1, 256>>>(dev_world, dev_camera, dev_framebuffer, height, width, states);
    cudaDeviceSynchronize();
    cudaMemcpy(framebuffer, dev_framebuffer, size * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaFree(dev_framebuffer);
    cudaFree(dev_world);
    cudaFree(dev_camera);
}
