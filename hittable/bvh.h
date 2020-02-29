#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "hittable.h"
#include "../random.h"

struct BVHNode : public Hittable
{
    __device__ BVHNode() {}
    __device__ BVHNode(Hittable **l, int n, float time_0, float time_1, curandState_t* state);
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AABB* box) const;
    Hittable *left;
    Hittable *right;
    AABB box;
};