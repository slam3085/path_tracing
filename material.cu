#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "material.h"
#include "random.h"
#include "onb.h"
#define M_PI 3.14159265358979323846f

__device__ bool Lambertian::scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, float& pdf, curandState_t* state) const
{
    ONB uvw;
    uvw.build_from_w(rec->normal);
    vec3 direction = uvw.local(random_cosine_direction(state));
    scattered->origin = rec->p;
    scattered->direction = direction.unit_vector();
    *attenuation = albedo->value(rec->u, rec->v, rec->p);
    pdf = dot(uvw.w, scattered->direction) / M_PI;
    return true;
}

__device__ float Lambertian::scattering_pdf(Ray* ray_in, HitRecord* rec, Ray* scattered) const
{
    float cosine = dot(rec->normal, scattered->direction.unit_vector());
    if(cosine < 0.0f)
        return 0.0f;
    return cosine / M_PI;
}

__device__ vec3 reflect(const vec3& v, const vec3& n)
{
    return v - n * 2.0f * dot(v, n);
}

__device__ bool Metal::scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, float& pdf, curandState_t* state) const
{
    scattered->origin = rec->p;
    scattered->direction = reflect(ray_in->direction.unit_vector(), rec->normal) + random_in_unit_sphere(state) * fuzziness;
    *attenuation = albedo;
    return dot(scattered->direction, rec->normal) > 0.0f;
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3* refracted)
{
    vec3 uv = v.unit_vector();
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if(discriminant > 0.0f)
    {
        *refracted = (uv - n * dt) * ni_over_nt - n * sqrtf(discriminant);
        return true;
    }
    return false;
}

__device__ float schlick(float cosine, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 *= r0;
    return r0 + (1.0f - r0) * (1.0f - cosine) * (1.0f - cosine) * (1.0f - cosine) * (1.0f - cosine) * (1.0f - cosine);
}

__device__ bool Dielectric::scatter(Ray* ray_in, HitRecord* rec, vec3* attenuation, Ray* scattered, float& pdf, curandState_t* state) const
{
    vec3 outward_normal;
    float ni_over_nt;
    *attenuation = { 1.0f, 1.0f, 1.0f };
    vec3 refracted;
    float reflect_prob = 1.0f;
    float cosine;
    if(dot(ray_in->direction, rec->normal) > 0.0f)
    {
        outward_normal = -(rec->normal);
        ni_over_nt = ref_idx;
        cosine = dot(ray_in->direction, rec->normal) / ray_in->direction.length();
        cosine = sqrtf(1.0f - ref_idx * ref_idx * (1.0f - cosine * cosine));
    }
    else
    {
        outward_normal = rec->normal;
        ni_over_nt = 1.0f / ref_idx;
        cosine = -dot(ray_in->direction, rec->normal) / ray_in->direction.length();
    }
    if(refract(ray_in->direction, outward_normal, ni_over_nt, &refracted))
        reflect_prob = schlick(cosine, ref_idx);
    scattered->origin = rec->p;
    if(random_float(state) < reflect_prob)
        scattered->direction = reflect(ray_in->direction, rec->normal);
    else
        scattered->direction = refracted;
    return true;
}