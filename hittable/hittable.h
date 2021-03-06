#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../vec3.h"
#include "../ray.h"
#include "../random.h"
#include "aabb.h"

struct Material;

struct HitRecord
{
    float t;
    vec3 p;
    vec3 normal;
    Material* material;
    float u;
    float v;
};

struct Hittable
{
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const = 0;
    __device__ virtual bool bounding_box(float t0, float t1, AABB* box) const = 0;
    __device__ virtual float pdf_value(const vec3& origin, const vec3& direction) const { return 0.0f; }
    __device__ virtual vec3 random(const vec3& origin, curandState_t* state) const { return vec3(1.0f, 0.0f, 0.0f); }
};

struct HittableList : public Hittable
{
    Hittable** list;
    int size;
    __device__ HittableList() {}
    __device__ HittableList(Hittable** l, int s) : list(l), size(s) {}
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AABB* box) const;
};