#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "ray.h"

struct HitRecord
{
    float t;
    vec3 p;
    vec3 normal;
};

struct Hittable
{
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const = 0;
};