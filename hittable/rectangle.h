#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../vec3.h"
#include "../ray.h"
#include "../random.h"
#include "hittable.h"


struct XYRect : public Hittable
{
    float x_0, x_1, y_0, y_1, k, n;
    Material* material;
    __device__ __host__ XYRect() {}
    __device__ __host__ XYRect(float _x_0, float _x_1, float _y_0, float _y_1, float _k, float _n, Material* m) :
        x_0(_x_0), x_1(_x_1), y_0(_y_0), y_1(_y_1), k(_k), n(_n), material(m) {}
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AABB* box) const;
};

struct XZRect : public Hittable
{
    float x_0, x_1, z_0, z_1, k, n;
    Material* material;
    __device__ __host__ XZRect() {}
    __device__ __host__ XZRect(float _x_0, float _x_1, float _z_0, float _z_1, float _k, float _n, Material* m) :
        x_0(_x_0), x_1(_x_1), z_0(_z_0), z_1(_z_1), k(_k), n(_n), material(m) {}
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AABB* box) const;
    __device__ virtual float pdf_value(const vec3& origin, const vec3& direction) const;
    __device__ virtual vec3 random(const vec3& origin, curandState_t* state) const;
};

struct YZRect : public Hittable
{
    float y_0, y_1, z_0, z_1, k, n;
    Material* material;
    __device__ __host__ YZRect() {}
    __device__ __host__ YZRect(float _y_0, float _y_1, float _z_0, float _z_1, float _k, float _n, Material* m) :
        y_0(_y_0), y_1(_y_1), z_0(_z_0), z_1(_z_1), k(_k), n(_n), material(m) {}
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AABB* box) const;
};

struct Box : public Hittable
{
    vec3 p_min, p_max;
    Hittable* list_ptr;
    __device__ Box() {}
    __device__ Box(const vec3& _p_min, const vec3& _p_max, Material* mat);
    __device__ virtual bool hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AABB* box) const;
};