#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"

struct Ray 
{
    vec3 origin, direction;
    __device__ __forceinline__ Ray(){}
    __device__ __forceinline__ Ray(const vec3& o, const vec3& d): origin(o), direction(d) {}
    __device__ __forceinline__ vec3 point_at_parameter(float t) const
    {
        return origin + direction * t;
    }
};