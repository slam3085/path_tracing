#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "camera.h"
#define _USE_MATH_DEFINES
#include <math.h>

__device__ Camera::Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect)
{
    vec3 u, v, w;
    float theta = vfov * M_PI / 180;
    float half_height = tan(theta / 2);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = (lookfrom - lookat).unit_vector();
    u = cross(vup, w).unit_vector();
    v = cross(w, u);
    lower_left_corner = origin - u * half_width - v * half_height - w;
    horizontal = u * 2 * half_width;
    vertical = v * 2 * half_height;
}

__device__ Ray Camera::get_ray(float s, float t) const
{
    return {
        origin,
        lower_left_corner + horizontal * s + vertical * t - origin
    };
}