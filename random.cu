#include "random.h"


__device__ float random_float(curandState_t* state)
{
    return float(curand(state) % 10000) / 10001.0;
}