#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "ray.h"
#include <math.h>
#define M_PI 3.14159265358979323846f


struct Camera 
{
    vec3 origin, lower_left_corner, horizontal, vertical;
    __device__ __forceinline__ Camera() {}
    __device__ __forceinline__ Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect)
    {
        vec3 u, v, w;
        float theta = vfov * M_PI / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = (lookfrom - lookat).unit_vector();
        u = cross(vup, w).unit_vector();
        v = cross(w, u);
        lower_left_corner = origin - u * half_width - v * half_height - w;
        horizontal = u * 2.0f * half_width;
        vertical = v * 2.0f * half_height;
    }
    __device__ __forceinline__ Ray get_ray(float s, float t) const
    {
        return Ray(
        origin,
        lower_left_corner + horizontal * s + vertical * t - origin
        );
    }
};