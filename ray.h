#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"

struct Ray 
{
    vec3 origin, direction;
    __device__ vec3 point_at_parameter(float t) const;
};