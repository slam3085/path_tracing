#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "math.h"

struct Texture
{
    __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

struct ConstantTexture : public Texture
{
    __device__ ConstantTexture() {}
    __device__ ConstantTexture(vec3 c) : color(c) {}
    __device__ virtual vec3 value(float u, float v, const vec3& p) const
    {
        return color;
    }
    vec3 color;
};

struct CheckerTexture : public Texture
{
    __device__ CheckerTexture() {}
    __device__ CheckerTexture(Texture* e, Texture* o) : even(e), odd(o) {}
    __device__ virtual vec3 value(float u, float v, const vec3& p) const
    {
        float sines = sin(5.0f * p.X) * sin(5.0f * p.Y) * sin(5.0f * p.Z);
        if (sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }
    Texture* even;
    Texture* odd;
};

struct ImageTexture : public Texture
{
    __device__ ImageTexture() {}
    __device__ ImageTexture(unsigned char* pixels, int A, int B): data(pixels), nx(A), ny(B) {}
    __device__ virtual vec3 value(float u, float v, const vec3& p) const
    {
        int i = u * nx;
        int j = (1.0f - v) * ny - 0.001f;
        if (i < 0) i = 0;
        if (j < 0) j = 0;
        if (i > nx - 1) i = nx - 1;
        if (j > ny - 1) j = ny - 1;
        float r = int(data[3 * i + 3 * nx * j]) / 255.0;
        float g = int(data[3 * i + 3 * nx * j + 1]) / 255.0;
        float b = int(data[3 * i + 3 * nx * j + 2]) / 255.0;
        return vec3(r * r, g * g, b * b);
    }
    unsigned char* data;
    int nx, ny;
};