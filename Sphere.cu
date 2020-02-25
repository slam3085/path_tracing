#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "sphere.h"

__device__ bool Sphere::hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const
{
    vec3 oc = ray->origin - center;
    float a = dot(ray->direction, ray->direction);
    float b = 2.0 * dot(oc, ray->direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant > 0)
    {
        float temp = (-b - sqrt(discriminant)) / 2.0 / a;
        if (temp < t_max && temp > t_min)
        {
            rec->t = temp;
            rec->p = ray->point_at_parameter(rec->t);
            rec->normal = (rec->p - center).unit_vector();
            return true;
        }
        temp = (-b + sqrt(discriminant)) / 2.0 / a;
        if (temp < t_max && temp > t_min)
        {
            rec->t = temp;
            rec->p = ray->point_at_parameter(rec->t);
            rec->normal = (rec->p - center).unit_vector();
            return true;
        }
    }
    return false;
}