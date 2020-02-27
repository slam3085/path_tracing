#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "ray.h"

struct Material;

struct HitRecord
{
    float t;
    vec3 p;
    vec3 normal;
    Material* material;
};

struct Hittable
{
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const = 0;
};

struct HittableList : public Hittable
{
    Hittable** list;
    int size;
    __device__ HittableList() {}
    __device__ HittableList(Hittable** l, int s) : list(l), size(s) {}
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const;
};