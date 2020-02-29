#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "hittable.h"

__device__ float ffmin(float a, float b);
__device__ float ffmax(float a, float b);

__device__  bool HittableList::hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const
{
    HitRecord temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for (int i = 0; i < size; i++) 
    {
        if(list[i]->hit(ray, t_min, closest_so_far, &temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            *rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ AABB surrounding_box(AABB* box_1, AABB* box_2)
{
    vec3 small = {
        ffmin(box_1->_min.X, box_2->_min.X),
        ffmin(box_1->_min.Y, box_2->_min.Y),
        ffmin(box_1->_min.Z, box_2->_min.Z)
    };
    vec3 big = {
        ffmax(box_1->_max.X, box_2->_max.X),
        ffmax(box_1->_max.Y, box_2->_max.Y),
        ffmax(box_1->_max.Z, box_2->_max.Z)
    };
    return AABB(small, big);
}

__device__ bool HittableList::bounding_box(float t0, float t1, AABB* box) const
{
    if(size < 1)
        return false;
    AABB temp_box;
    bool first_true = list[0]->bounding_box(t0, t1, &temp_box);
    if(!first_true)
        return false;
    else
        *box = temp_box;
    for(int i = 1; i < size; i++)
        if(list[i]->bounding_box(t0, t1, &temp_box))
            *box = surrounding_box(box, &temp_box);
        else
            return false;
    return true;
}