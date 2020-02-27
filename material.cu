#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "material.h"

__device__ bool Lambertian::scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, curandState_t* state) const
{
    vec3 target = rec->p + rec->normal + random_unit_in_sphere(state);
    scattered->origin = rec->p;
    scattered->direction = target - rec->p;
    *attenuation = albedo;
    return true;
}

__device__ vec3 reflect(const vec3& v, const vec3& n)
{
    return v - n * 2 * dot(v, n);
}

__device__ bool Metal::scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, curandState_t* state) const
{
    vec3 reflected = reflect(ray_in->direction.unit_vector(), rec->normal);
    scattered->origin = rec->p;
    scattered->direction = reflected + random_unit_in_sphere(state) * fuzziness;
    *attenuation = albedo;
    return dot(scattered->direction, rec->normal) > 0;
}