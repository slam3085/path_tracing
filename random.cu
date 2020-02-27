#include "random.h"

__device__ float random_float(curandState_t* state)
{
    //curand_uniform returns values from (0.0, 1.0]
    float tmp = curand_uniform(state);
    if(tmp >= 1.0)
        return 0.0;
    return tmp;
}

__device__ vec3 random_unit_in_sphere(curandState_t* state)
{
    vec3 p;
    while(true)
    {
        p.X = 2.0 * random_float(state) - 1.0;
        p.Y = 2.0 * random_float(state) - 1.0;
        p.Z = 2.0 * random_float(state) - 1.0;
        if(p.squared_length() < 1.0)
            return p;
    }
}