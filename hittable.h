#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"

struct HitRecord
{
    float t;
    vec3 p;
    vec3 normal;
};