#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bvh.h"

__device__ AABB surrounding_box(AABB* box_1, AABB* box_2);

__device__ void not_qsort(Hittable** l, int n, int sort_by)
{
    for(int i = 0; i < n - 1; i++)
        for(int j = i + 1; j < n; j++)
        {
            AABB box_left, box_right;
            l[i]->bounding_box(0, 0, &box_left);
            l[j]->bounding_box(0, 0, &box_right);
            if(box_left._min[sort_by] > box_right._min[sort_by])
            {
                Hittable* tmp = l[j];
                l[j] = l[i];
                l[i] = tmp;
            }
        }
}

__device__ BVHNode::BVHNode(Hittable** l, int n, float time_0, float time_1, curandState_t* state)
{
    int axis = int(3.0f * random_float(state));
    if (axis == 0)
        not_qsort(l, n, 0);
    else if (axis == 1)
        not_qsort(l, n, 1);
    else if (axis == 2)
        not_qsort(l, n, 2);
    if (n == 1)
    {
        left = right = l[0];
    }
    else if (n == 2)
    {
        left = l[0];
        right = l[1];
    }
    else
    {
        left = new BVHNode(l, n / 2, time_0, time_1, state);
        right = new BVHNode(l + n/2, n - n / 2, time_0, time_1, state);
    }
    AABB box_left, box_right;
    left->bounding_box(time_0, time_1, &box_left);
    right->bounding_box(time_0, time_1, &box_right);
    box = surrounding_box(&box_left, &box_right);
}

__device__ bool BVHNode::hit(Ray* ray, float t_min, float t_max, HitRecord* rec) const
{
    if(box.hit(ray, t_min, t_max))
    {
        HitRecord left_rec, right_rec;
        bool hit_left = left->hit(ray, t_min, t_max, &left_rec);
        bool hit_right = right->hit(ray, t_min, t_max, &right_rec);
        if(hit_left && hit_right)
        {
            if(left_rec.t < right_rec.t)
                *rec = left_rec;
            else
                *rec = right_rec;
            return true;
        }
        else if(hit_left)
        {
            *rec = left_rec;
            return true;
        }
        else if(hit_right)
        {
            *rec = right_rec;
            return true;
        }
        return false;
    }
    return false;
}

__device__ bool BVHNode::bounding_box(float t0, float t1, AABB* b) const
{
    *b = box;
    return true;
}