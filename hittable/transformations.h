#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "hittable.h"


struct Translate : public Hittable
{
    __device__ Translate(Hittable* h, const vec3& o): hittable(h), offset(o) {}
    __device__ __forceinline__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const;
    __device__ __forceinline__ virtual bool bounding_box(float t0, float t1, AABB* box) const;
    Hittable* hittable;
    vec3 offset;
};

struct RotateY : public Hittable
{
    __device__ RotateY(Hittable* h, float angle);
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AABB* box) const;
    Hittable* hittable;
    float sin_theta;
    float cos_theta;
    bool hasbox;
    AABB bbox;
};