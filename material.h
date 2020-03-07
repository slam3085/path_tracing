#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ray.h"
#include "hittable/hittable.h"
#include "random.h"
#include "texture.h"

struct Material
{
    __device__ virtual bool scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, float& pdf, curandState_t* state) const
    {
        return false;
    }
    __device__ virtual float scattering_pdf(Ray* ray_in, HitRecord* rec, Ray* scattered) const
    {
        return 0.0f;
    }
    __device__ virtual vec3 emitted(Ray* ray_in, HitRecord* rec, float u, float v, const vec3& p) const
    {
        return { 0.0f, 0.0f, 0.0f };
    }
};

struct Lambertian : public Material
{
    __device__ Lambertian(Texture* a): albedo(a) {}
    __device__ virtual bool scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, float& pdf, curandState_t* state) const;
    __device__ float scattering_pdf(Ray * ray_in, HitRecord * rec, Ray * scattered) const;
    Texture* albedo;
};

struct Metal : public Material
{
    __device__ Metal(const vec3& a, float f) : albedo(a), fuzziness(f) {}
    __device__ virtual bool scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, float& pdf, curandState_t* state) const;
    vec3 albedo;
    float fuzziness;
};

struct Dielectric : public Material
{
    __device__ Dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, float& pdf, curandState_t* state) const;
    float ref_idx;
};

struct DiffuseLight : public Material
{
    __device__ DiffuseLight(Texture* e) : emit(e) {}
    __device__ virtual bool scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, float& pdf, curandState_t* state) const
    {
        return false;
    }
    __device__ virtual vec3 emitted(Ray* ray_in, HitRecord* rec, float u, float v, const vec3& p) const
    {
        if(dot(rec->normal, ray_in->direction) < 0.0f)
            return emit->value(u, v, p);
        return vec3(0.0f, 0.0f, 0.0f);
    }
    Texture *emit;
};