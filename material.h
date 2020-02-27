#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ray.h"
#include "hittable.h"
#include "random.h"

struct Material
{
    __device__ virtual bool scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, curandState_t* state) const = 0;
};

struct Lambertian : public Material
{
    __device__ Lambertian(const vec3& a): albedo(a) {}
    __device__ virtual bool scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, curandState_t* state) const;
    vec3 albedo;
};

struct Metal : public Material
{
    __device__ Metal(const vec3& a, float f) : albedo(a), fuzziness(f) {}
    __device__ virtual bool scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, curandState_t* state) const;
    vec3 albedo;
    float fuzziness;
};