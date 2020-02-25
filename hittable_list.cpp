#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "hittable_list.h"


bool HittableList::hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const 
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