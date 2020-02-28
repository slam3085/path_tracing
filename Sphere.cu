#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "sphere.h"

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
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec->t = temp;
            rec->p = ray->point_at_parameter(rec->t);
            rec->normal = (rec->p - center) / radius;
            rec->material = material;
            return true;
        }
    }
    return false;
}