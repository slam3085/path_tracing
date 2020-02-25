#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "ray.h"
#include "hittable.h"

struct Sphere : public Hittable
{
    vec3 center;
    float radius;
    __device__ __host__ Sphere() {}
    __device__ __host__ Sphere(vec3 c, float r): center(c), radius(r) {}
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const;
};