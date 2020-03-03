#include "math.h"
#include "transformations.h"
# define M_PI 3.14159265358979323846f

__device__ bool Translate::hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const
{
    Ray moved_r = { ray->origin - offset, ray->direction };
    if(hittable->hit(&moved_r, t_min, t_max, rec))
    {
        rec->p += offset;
        return true;
    }
    return false;
}

__device__ bool Translate::bounding_box(float t0, float t1, AABB* box) const
{
    if(hittable->bounding_box(t0, t1, box))
    {
        *box = AABB(box->_min + offset, box->_max + offset);
        return true;
    }
    return false;
}

__device__ RotateY::RotateY(Hittable* h, float angle) : hittable(h)
{
    float radians = (M_PI / 180.0f) * angle;
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    hasbox = hittable->bounding_box(0, 1, &bbox);
    vec3 min = { 1E38f, 1E38f, 1E38f };
    vec3 max = {-1E38f, -1E38f , -1E38f };
    vec3 tester;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                float x = i * bbox._max.X + (1 - i) * bbox._min.X;
                float y = j * bbox._max.Y + (1 - j) * bbox._min.Y;
                float z = k * bbox._max.Z + (1 - k) * bbox._min.Z;
                float newx = cos_theta * x + sin_theta * z;
                float newz = -sin_theta * x + cos_theta * z;
                tester.X = newx; tester.Y = y; tester.Z = newz;
                for(int c = 0; c < 3; c++)
                {
                    if(tester[c] > max[c])
                        max[c] = tester[c];
                    if(tester[c] < min[c])
                        min[c] = tester[c];
                }
            }
        }
    }
    bbox = AABB(min, max);
}

__device__ bool RotateY::hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const
{
    vec3 origin = ray->origin;
    vec3 direction = ray->direction;
    origin.X = cos_theta * ray->origin.X - sin_theta * ray->origin.Z;
    origin.Z = sin_theta * ray->origin.X + cos_theta * ray->origin.Z;
    direction.X = cos_theta * ray->direction.X - sin_theta * ray->direction.Z;
    direction.Z = sin_theta * ray->direction.X + cos_theta * ray->direction.Z;
    Ray rotated_r = { origin, direction };
    if(hittable->hit(&rotated_r, t_min, t_max, rec))
    {
        vec3 p = rec->p;
        vec3 normal = rec->normal;
        p.X = cos_theta * rec->p.X + sin_theta * rec->p.Z;
        p.Z = -sin_theta * rec->p.X + cos_theta * rec->p.Z;
        normal.X = cos_theta * rec->normal.X + sin_theta * rec->normal.Z;
        normal.Z = -sin_theta * rec->normal.X + cos_theta * rec->normal.Z;
        rec->p = p;
        rec->normal = normal;
        return true;
    }
    else
        return false;
}

__device__ bool RotateY::bounding_box(float t0, float t1, AABB* box) const
{
    *box = bbox;
    return hasbox;
}