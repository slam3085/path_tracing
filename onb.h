#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "math.h"

struct ONB
{
    vec3 u, v, w;
    __device__ __forceinline__ ONB() {}
    __device__ __forceinline__ vec3 operator[](int i) const
    {
        switch (i)
        {
        case 0:
            return u;
        case 1:
            return v;
        case 2:
            return w;
        }
    }
    __device__ __forceinline__ vec3 local(float a, float b, float c) const
    {
        return u * a + v * b + w * c;
    }
    __device__ __forceinline__ vec3 local(const vec3& a) const
    {
        return u * a.X + v * a.Y + w * a.Z;
    }
    __device__ __forceinline__ void build_from_w(const vec3& n)
    {
        w = n.unit_vector();
        vec3 a = (fabs(w.X) > 0.9f) ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
        v = cross(w, a).unit_vector();
        u = cross(w, v);
    }
};