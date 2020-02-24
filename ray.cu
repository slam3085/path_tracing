#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "ray.h"

__device__ vec3 Ray::point_at_parameter(float t) const
{
    return origin + direction * t;
}