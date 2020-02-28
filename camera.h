#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "ray.h"

struct Camera 
{
    vec3 origin, lower_left_corner, horizontal, vertical;
    __device__ Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect);
    __device__ Ray get_ray(float u, float v) const;
};