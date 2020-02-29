#include "aabb.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float ffmin(float a, float b)
{
    return a < b ? a : b;
}

__device__ float ffmax(float a, float b)
{
    return a > b ? a : b;
}

__device__ bool AABB::hit(Ray* ray, float t_min, float t_max) const
{
    for(int a = 0; a < 3; a++)
    {
        float invD = 1.0f / ray->direction[a];
        float t0 = (_min[a] - ray->origin[a]) * invD;
        float t1 = (_max[a] - ray->origin[a]) * invD;
        if(invD < 0.0f)
        {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if(t_max <= t_min)
            return false;
    }
    return true;
}