#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "path_tracing.h"
#include "ray.h"
#include "hittable/sphere.h"
#include "hittable/hittable.h"
#include "hittable/rectangle.h"
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
        
        Material* red = new Lambertian(new ConstantTexture({ 0.65, 0.05, 0.05 }));
        Material* white = new Lambertian(new ConstantTexture({ 0.73, 0.73, 0.73 }));
        Material* green = new Lambertian(new ConstantTexture({ 0.12, 0.45, 0.15 }));
        Material* light = new DiffuseLight(new ConstantTexture({ 15, 15, 15 }));
        int n = 8;
        Hittable** list = new Hittable*[n];
        list[0] = new YZRect(0, 555, 0, 555, 555, -1, green);
        list[1] = new YZRect(0, 555, 0, 555, 0, 1, red);
        list[2] = new XZRect(213, 343, 227, 332, 554, 1, light);
        list[3] = new XZRect(0, 555, 0, 555, 555, -1, white);
        list[4] = new XZRect(0, 555, 0, 555, 0, 1, white);
        list[5] = new XYRect(0, 555, 0, 555, 555, -1, white);
        list[6] = new Box({ 130, 0, 65 }, { 295, 165, 230}, white);
        list[7] = new Box({ 265, 0, 295 }, { 430, 330, 460 }, white);
        *dev_world = new HittableList(list, n);
        *dev_camera = new Camera({ 278, 278, -800 }, { 278,278,0 }, { 0, 1, 0 }, 40, float(width) / float(height));
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

__global__ void path_tracing_kernel(Hittable** dev_world, Camera** dev_camera, vec3* dev_framebuffer, int height, int width, curandState_t* states, int rays_per_pixel)
{
    int size = width * height;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        int i = idx % width;
        int j = idx / width;
        vec3 col = { 0.0f, 0.0f, 0.0f };
        for (int s = 0; s < rays_per_pixel; s++)
        {
            float u = (float(i) + random_float(&states[idx])) / float(width);
            float v = (float(j) + random_float(&states[idx])) / float(height);
            Ray ray = (*dev_camera)->get_ray(u, v);
            col += color(&ray, dev_world, &states[idx]);
        }
        //col /= float(rays_per_pixel);
        //col.X = sqrt(col.X);
        //col.Y = sqrt(col.Y);
        //col.Z = sqrt(col.Z);
        dev_framebuffer[idx] = col;
    }
}

void path_tracing_with_cuda(std::string filename, int height, int width)
{
    const int n_threads = 512;
    cudaSetDevice(0);
    // earth
    int nx, ny, nn;
    unsigned char* tex_data = stbi_load("textures/stickor.jpg", &nx, &ny, &nn, 0);
    unsigned char* dev_tex_data = 0;
    cudaMalloc((void**)&dev_tex_data, 3 * nx * ny * sizeof(unsigned char));
    cudaMemcpy(dev_tex_data, tex_data, 3 * nx * ny * sizeof(unsigned char), cudaMemcpyHostToDevice);
    //framebuffer
    int size = width * height;
    vec3* framebuffer = (vec3*)malloc(size * sizeof(vec3));
    vec3* dev_framebuffer = 0;
    cudaMalloc((void**)&dev_framebuffer, size * sizeof(vec3));
    //camera
    Camera** dev_camera = 0;
    cudaMalloc(&dev_camera, sizeof(Camera**));
    //curand
    curandState_t* states = 0;
    cudaMalloc((void**)&states, size * sizeof(curandState_t));
    init_curand<<<size / n_threads + 1, n_threads>>>(states, height, width);
    //world
    Hittable** dev_world = 0;
    cudaMalloc(&dev_world, sizeof(Hittable**));
    init_common <<<1,1>>>(dev_world, dev_tex_data, nx, ny, dev_camera, height, width, states);
    //tracing
    int rays_per_pixel = 128;
    int total_rays_per_pixel = 0;
    while(total_rays_per_pixel < 1000)
    {
        path_tracing_kernel<<<size / n_threads + 1, n_threads >>>(dev_world, dev_camera, dev_framebuffer, height, width, states, rays_per_pixel);
        cudaDeviceSynchronize();
        total_rays_per_pixel += rays_per_pixel;
        cudaMemcpy(framebuffer, dev_framebuffer, size * sizeof(vec3), cudaMemcpyDeviceToHost);
        render(filename, framebuffer, height, width, total_rays_per_pixel);
        rays_per_pixel = total_rays_per_pixel;
        std::cout << total_rays_per_pixel << " iterations passed\n";
    }
    cudaFree(dev_framebuffer);
    cudaFree(dev_world);
    cudaFree(dev_camera);
    free(framebuffer);
}
