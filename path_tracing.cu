#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "path_tracing.h"
#include "ray.h"
#include "hittable/sphere.h"
#include "hittable/hittable.h"
#include "random.h"
#include "camera.h"
#include "material.h"
#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image.h"
#include <iostream>

__global__ void init_common(Hittable** dev_world, unsigned char* dev_tex_data, int nx, int ny, Camera** dev_camera, int height, int width, curandState_t* states)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        int n = 5;
        Hittable** list = new Hittable*[n];
        Texture* checker = new CheckerTexture(
            new ConstantTexture({ 0.0, 0.0, 0.0 }),
            new ConstantTexture({ 0.6, 0.6, 0.6 })
        );
        list[0] = new Sphere({ 0, -1000, 0 }, 1000, new Lambertian(checker));
        list[1] = new Sphere({ 2, 1, -1 }, 1.0, new Metal({ 1.0, 1.0, 1.0 }, 0.0));
        list[2] = new Sphere({ 2, 1, 2 }, 1.0, new Lambertian(new ImageTexture(dev_tex_data, nx, ny)));
        list[3] = new Sphere({ 4, 1, 0.5 }, 1.0, new Dielectric(1.5));
        list[4] = new Sphere({ 8, 8, -4 }, 1.0, new DiffuseLight(new ConstantTexture({ 10.0f, 10.0f, 10.0f })));
        *dev_world = new HittableList(list, n);
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
    HitRecord rec;
    const int depth = 50;
    int iters = 0;
    vec3 attenuation_stack[depth + 1];
    vec3 emitted_stack[depth + 1];
    int filled = 0;
    Ray scattered;
    while((*dev_world)->hit(ray, 0.001f, 1E38f, &rec))
    {
        emitted_stack[iters] = rec.material->emitted(rec.u, rec.v, rec.p);
        filled += 1;
        if(iters < depth && rec.material->scatter(ray, &rec, &attenuation_stack[iters], &scattered, state))
        {
            ray = &scattered;
            iters += 1;
        }
        else
        {
            attenuation_stack[iters] = { 0.0f, 0.0f, 0.0f };
            break;
        }
    }
    vec3 col = { 0.0f, 0.0f, 0.0f };
    for(int i = filled - 1; i >= 0; i--)
        col = emitted_stack[i] + attenuation_stack[i] * col;
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
    // earth
    int nx, ny, nn;
    unsigned char* tex_data = stbi_load("textures/stickor.jpg", &nx, &ny, &nn, 0);
    unsigned char* dev_tex_data = 0;
    cudaMalloc((void**)&dev_tex_data, 3 * nx * ny * sizeof(unsigned char));
    cudaMemcpy(dev_tex_data, tex_data, 3 * nx * ny * sizeof(unsigned char), cudaMemcpyHostToDevice);
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
    init_curand<<<size / 256 + 1, 512>>>(states, height, width);
    //world
    Hittable** dev_world = 0;
    cudaMalloc(&dev_world, sizeof(Hittable**));
    init_common <<<1,1>>>(dev_world, dev_tex_data, nx, ny, dev_camera, height, width, states);
    //tracing
    path_tracing_kernel<<<size / 256 + 1, 512>>>(dev_world, dev_camera, dev_framebuffer, height, width, states);
    cudaDeviceSynchronize();
    cudaMemcpy(framebuffer, dev_framebuffer, size * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaFree(dev_framebuffer);
    cudaFree(dev_world);
    cudaFree(dev_camera);
}
