#include "random.h"

__device__ float random_float(curandState_t* state)
{
    //curand_uniform returns values from (0.0, 1.0]
    return 1.0f - curand_uniform(state);
}

__device__ vec3 random_unit_in_sphere(curandState_t* state)
{
    vec3 p;
    while(true)
    {
        p.X = 2.0f * random_float(state) - 1.0f;
        p.Y = 2.0f * random_float(state) - 1.0f;
        p.Z = 2.0f * random_float(state) - 1.0f;
        if(p.squared_length() < 1.0f)
            return p;
    }
}