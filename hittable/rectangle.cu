#include "rectangle.h"

__device__ bool XYRect::hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const
{
    float t = (k - ray->origin.Z) / ray->direction.Z;
    if (t < t_min || t > t_max)
        return false;
    float x = ray->origin.X + t * ray->direction.X;
    float y = ray->origin.Y + t * ray->direction.Y;
    if (x < x_0 || x > x_1 || y < y_0 || y > y_1)
        return false;
    rec->u = (x - x_0) / (x_1 - x_0);
    rec->v = (y - y_0) / (y_1 - y_0);
    rec->t = t;
    rec->material = material;
    rec->p = ray->point_at_parameter(t);
    rec->normal = { 0.0f, 0.0f, n };
    return true;
}

__device__ bool XYRect::bounding_box(float t0, float t1, AABB* box) const
{
    *box = AABB({ x_0, y_0, k - 0.0001f }, { x_1, y_1, k + 0.0001f });
    return true;
}

__device__ bool XZRect::hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const
{
    float t = (k - ray->origin.Y) / ray->direction.Y;
    if (t < t_min || t > t_max)
        return false;
    float x = ray->origin.X + t * ray->direction.X;
    float z = ray->origin.Z + t * ray->direction.Z;
    if (x < x_0 || x > x_1 || z < z_0 || z > z_1)
        return false;
    rec->u = (x - x_0) / (x_1 - x_0);
    rec->v = (z - z_0) / (z_1 - z_0);
    rec->t = t;
    rec->material = material;
    rec->p = ray->point_at_parameter(t);
    rec->normal = { 0.0f, n, 0.0f };
    return true;
}

__device__ bool XZRect::bounding_box(float t0, float t1, AABB* box) const
{
    *box = AABB({ x_0, k - 0.0001f, z_0 }, { x_1, k + 0.0001f, z_1,  });
    return true;
}

__device__ bool YZRect::hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const
{
    float t = (k - ray->origin.X) / ray->direction.X;
    if (t < t_min || t > t_max)
        return false;
    float y = ray->origin.Y + t * ray->direction.Y;
    float z = ray->origin.Z + t * ray->direction.Z;
    if (y < y_0 || y > y_1 || z < z_0 || z > z_1)
        return false;
    rec->u = (y - y_0) / (y_1 - y_0);
    rec->v = (z - z_0) / (z_1 - z_0);
    rec->t = t;
    rec->material = material;
    rec->p = ray->point_at_parameter(t);
    rec->normal = { n, 0.0f, 0.0f };
    return true;
}

__device__ bool YZRect::bounding_box(float t0, float t1, AABB* box) const
{
    *box = AABB({ k - 0.0001f, y_0, z_0 }, { k + 0.0001f, y_1, z_1 });
    return true;
}

__device__ Box::Box(const vec3& _p_min, const vec3& _p_max, Material* mat) : p_min(_p_min), p_max(_p_max)
{
    Hittable** list = new Hittable*[6];
    list[0] = new XYRect(p_min.X, p_max.X, p_min.Y, p_max.Y, p_max.Z, 1, mat);
    list[1] = new XYRect(p_min.X, p_max.X, p_min.Y, p_max.Y, p_min.Z, -1, mat);
    list[2] = new XZRect(p_min.X, p_max.X, p_min.Z, p_max.Z, p_max.Y, 1, mat);
    list[3] = new XZRect(p_min.X, p_max.X, p_min.Z, p_max.Z, p_min.Y, -1, mat);
    list[4] = new YZRect(p_min.Y, p_max.Y, p_min.Z, p_max.Z, p_max.X, 1, mat);
    list[5] = new YZRect(p_min.Y, p_max.Y, p_min.Z, p_max.Z, p_min.X, -1, mat);
    list_ptr = new HittableList(list, 6);
}

__device__ bool Box::hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const
{
    return list_ptr->hit(ray, t_min, t_max, rec);
}

__device__ bool Box::bounding_box(float t0, float t1, AABB* box) const
{
    *box = AABB(p_min, p_max);
    return true;
}