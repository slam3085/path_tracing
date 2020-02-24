#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ray.h"
#include "vec3.h"

__device__ vec3 Ray::point_at_parameter(float t)
{
    return origin + direction * t;
}