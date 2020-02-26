#include "random.h"


__device__ float random_float(curandState_t* state)
{
    return float(curand(state) % 1000000000) / float(1000000001);
}

__device__ vec3 random_unit_in_sphere(curandState_t* state)
{
    vec3 p;
    do
    {
        p.X = 2.0 * random_float(state) - 1.0;
        p.Y = 2.0 * random_float(state) - 1.0;
        p.Z = 2.0 * random_float(state) - 1.0;
    } while (p.squared_length() >= 1.0);
    return p;
}