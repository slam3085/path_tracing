#include <SFML/Graphics.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "path_tracing.h"
#include "ray.h"
#include "hittable/sphere.h"
#include "hittable/hittable.h"
#include "hittable/rectangle.h"
#include "hittable/transformations.h"
#include "random.h"
#include "pdf.h"
#include "camera.h"
#include "material.h"
#include <float.h>
#include <iostream>
#include <chrono>
#define NTHREADS 128
#define MAX_DEPTH 15

__global__ void init_common(Hittable** dev_world, Hittable** light_shapes, Camera** dev_camera, int height, int width, curandState_t* states)
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
        list[2] = new XZRect(213, 343, 227, 332, 554, -1, light);
        list[3] = new XZRect(0, 555, 0, 555, 555, -1, white);
        list[4] = new XZRect(0, 555, 0, 555, 0, 1, white);
        list[5] = new XYRect(0, 555, 0, 555, 555, -1, white);
        list[6] = new Translate(new RotateY(new Box({ 0, 0, 0 }, { 165, 165, 165 }, white), -18), { 130,0,65 });
        list[7] = new Translate(new RotateY(new Box({ 0, 0, 0 }, { 165, 330, 165 }, white), 15), { 265,0,295 });
        *dev_world = new HittableList(list, n);
        light_shapes[0] = new XZRect(213, 343, 227, 332, 554, 1, 0);
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

__device__ vec3 color(Ray* ray, Hittable** dev_world, MixturePDF* pdf, int depth, HitRecord& rec, curandState_t* state)
{
    if((*dev_world)->hit(ray, 0.001f, FLT_MAX, &rec))
    {
        Ray scattered;
        vec3 emitted = rec.material->emitted(ray, &rec, rec.u, rec.v, rec.p);
        vec3 albedo;
        float pdf_val;
        if (depth < MAX_DEPTH && rec.material->scatter(ray, &rec, &albedo, &scattered, pdf_val, state))
        {
            ((HittablePDF*)pdf->p_0)->origin = rec.p;
            ((CosinePDF*)pdf->p_1)->uvw.build_from_w(rec.normal);
            scattered.origin = rec.p;
            scattered.direction = pdf->generate(state);
            pdf_val = pdf->value(scattered.direction);
            return emitted + albedo / pdf_val * rec.material->scattering_pdf(ray, &rec, &scattered) * color(&scattered, dev_world, pdf, depth + 1, rec, state);
        } 
        return emitted;
    }
    return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void path_tracing_kernel(Hittable** dev_world, Hittable** light_shapes, Camera** dev_camera,
    vec3* framebuffer, unsigned char* pixels, int height, int width, 
    curandState_t* states, int rays_per_pixel, int total_rays_per_pixel)
{
    int size = width * height;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;
    int i = idx % width;
    int j = idx / width;
    int t = threadIdx.x;
    __shared__ vec3 col[NTHREADS];
    col[t] = { 0.0f, 0.0f, 0.0f };
    __shared__ Ray ray[NTHREADS];
    __shared__ curandState_t local_state[NTHREADS];
    local_state[t] = states[idx];
    __shared__ HitRecord rec[NTHREADS];
    __shared__ PDF* p_0[NTHREADS];
    __shared__ PDF* p_1[NTHREADS];
    __shared__ MixturePDF* p[NTHREADS];
    p_0[t] = new HittablePDF(light_shapes[0], vec3(0, 0, 0));
    p_1[t] = new CosinePDF(vec3(0, 0, 0));
    p[t] = new MixturePDF(p_0[t], p_1[t]);
    //path tracing iterations
    for (int s = 0; s < rays_per_pixel; s++)
    {
        float u = (float(i) + random_float(&local_state[t])) / float(width);
        float v = (float(j) + random_float(&local_state[t])) / float(height);
        ray[t] = (*dev_camera)->get_ray(u, v);
        col[t] += de_nan(color(&ray[t], dev_world, p[t], 0, rec[t], &local_state[t]));
    }
    //calc color and put to global buffers
    framebuffer[idx] += col[t];
    col[t] = framebuffer[idx];
    float n_rays = float(total_rays_per_pixel + rays_per_pixel);
    int r = int(255.99f * sqrtf(col[t].X / n_rays));
    if(r > 255) r = 255;
    int g = int(255.99f * sqrtf(col[t].Y / n_rays));
    if(g > 255) g = 255;
    int b = int(255.99f * sqrtf(col[t].Z / n_rays));
    if(b > 255) b = 255;
    pixels[4 * ((height - 1 - j) * width + i)] = r;
    pixels[4 * ((height - 1 - j) * width + i) + 1] = g;
    pixels[4 * ((height - 1 - j) * width + i) + 2] = b;
    //copy from shared to global memory
    states[idx] = local_state[t];
    delete p_0[t];
    delete p_1[t];
    delete p[t];
}

void path_tracing_with_cuda(std::string filename, int height, int width)
{
    cudaSetDevice(0);
    //framebuffer
    int size = width * height;
    vec3* framebuffer = 0;
    cudaMalloc(&framebuffer, size * sizeof(vec3));
    unsigned char* pixels = 0;
    cudaMallocManaged(&pixels, 4 * size * sizeof(unsigned char));
    cudaMemset(pixels, 255, 4 * size * sizeof(unsigned char));
    //camera
    Camera** dev_camera = 0;
    cudaMalloc(&dev_camera, sizeof(Camera**));
    //curand
    curandState_t* states = 0;
    cudaMalloc((void**)&states, size * sizeof(curandState_t));
    init_curand<<<size / NTHREADS + 1, NTHREADS>>>(states, height, width);
    //world
    Hittable** dev_world = 0;
    cudaMalloc(&dev_world, sizeof(Hittable**));
    Hittable** light_shapes = 0;
    cudaMalloc(&light_shapes, sizeof(Hittable**));
    init_common<<<1,1>>>(dev_world, light_shapes, dev_camera, height, width, states);
    //SFML
    sf::RenderWindow window(sf::VideoMode(width, height), "path tracing");
    sf::Texture texture;
    texture.create(width, height);
    sf::Sprite sprite(texture);
    //tracing
    int rays_per_pixel = 100;
    int total_rays_per_pixel = 0;
    while(window.isOpen() && total_rays_per_pixel < 1000)
    {
        sf::Event event;
        while(window.pollEvent(event))
        {
            if(event.type == sf::Event::Closed)
                window.close();
        }
        auto start = std::chrono::steady_clock::now();
        path_tracing_kernel<<<size / NTHREADS + 1, NTHREADS>>>(dev_world, light_shapes, dev_camera, framebuffer, pixels, height, width, states, rays_per_pixel, total_rays_per_pixel);
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << diff << " ms passed; " << float(width * height * rays_per_pixel) / float(diff * 1000) << " Mrays/s" << std::endl;
        total_rays_per_pixel += rays_per_pixel;
        texture.update((sf::Uint8*)pixels);
        window.clear();
        window.draw(sprite);
        window.display();
    }
    sf::Image final_pic;
    final_pic.create(width, height, (sf::Uint8*)pixels);
    final_pic.saveToFile(filename);
    cudaFree(framebuffer);
    cudaFree(dev_world);
    cudaFree(dev_camera);
    cudaFree(pixels);
}
