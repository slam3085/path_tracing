#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../vec3.h"
#include "../ray.h"

struct AABB
{
    __device__ AABB() {}
    __device__ AABB(const vec3& a, const vec3& b) : _min(a), _max(b) {}
    __device__ bool hit(Ray* ray, float t_min, float t_max) const;
    vec3 _min, _max;
};