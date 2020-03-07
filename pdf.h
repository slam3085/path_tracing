#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "hittable/hittable.h"
#include "onb.h"
#include "random.h"
#define M_PI 3.14159265358979323846f


struct PDF
{
    __device__ PDF() {}
    __device__ virtual float value(const vec3&direction) const = 0;
    __device__ virtual vec3 generate(curandState_t* state) const = 0;
};

struct CosinePDF : public PDF
{
    ONB uvw;
    __device__ CosinePDF() {}
    __device__ CosinePDF(const vec3& w) { uvw.build_from_w(w); }
    __device__ virtual float value(const vec3&direction) const
    {
        float cosine = dot(direction.unit_vector(), uvw.w);
        if (cosine > 0.0f)
            return cosine / M_PI;
        else
            return 0.0f;
    }
    __device__ virtual vec3 generate(curandState_t* state) const
    {
        return uvw.local(random_cosine_direction(state));
    }
};

struct HittablePDF : public PDF
{
    vec3 origin;
    Hittable* hittable;
    __device__ HittablePDF() {}
    __device__ HittablePDF(Hittable* h, const vec3& o) : hittable(h), origin(o) {}
    __device__ virtual float value(const vec3& direction) const
    {
        return hittable->pdf_value(origin, direction);
    }
    __device__ virtual vec3 generate(curandState_t* state) const
    {
        return hittable->random(origin, state);
    }
};

struct MixturePDF : public PDF
{
    PDF* p_0;
    PDF* p_1;
    __device__ MixturePDF() {}
    __device__ MixturePDF(PDF* _p_0, PDF* _p_1) : p_0(_p_0), p_1(_p_1) {}
    __device__ virtual float value(const vec3& direction) const
    {
        return 0.5 * p_0->value(direction) + 0.5 *p_1->value(direction);
    }
    __device__ virtual vec3 generate(curandState_t* state) const
    {
        if(random_float(state) < 0.5)
            return p_0->generate(state);
        return p_1->generate(state);
    }
};