#include "cuda_runtime.h"
#include "device_launch_parameters.h"
# define M_PI 3.14159265358979323846f
#include "math.h"
#include "sphere.h"

__device__ void get_sphere_uv(const vec3& p, HitRecord* rec)
{
    float phi = atan2(p.Z, p.X);
    float theta = asin(p.Y);
    rec->u = 1.0f - (phi + M_PI) / (2.0f * M_PI);
    rec->v = (theta + M_PI / 2.0f) / M_PI;
}

__device__ bool Sphere::hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const
{
    vec3 oc = ray->origin - center;
    float a = dot(ray->direction, ray->direction);
    float b = dot(oc, ray->direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0.0f)
    {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec->t = temp;
            rec->p = ray->point_at_parameter(rec->t);
            rec->normal = (rec->p - center) / radius;
            rec->material = material;
            get_sphere_uv((rec->p - center) / radius, rec);
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec->t = temp;
            rec->p = ray->point_at_parameter(rec->t);
            rec->normal = (rec->p - center) / radius;
            rec->material = material;
            get_sphere_uv((rec->p - center) / radius, rec);
            return true;
        }
    }
    return false;
}

__device__ bool Sphere::bounding_box(float t0, float t1, AABB* box) const
{
    *box = AABB(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
    return true;
}