#include "random.h"
#define M_PI 3.14159265358979323846f

__device__ float random_float(curandState_t* state)
{
    //curand_uniform returns values from (0.0, 1.0]
    return 1.0f - curand_uniform(state);
}

__device__ vec3 random_in_unit_sphere(curandState_t* state)
{
    vec3 p(2.0f * random_float(state) - 1.0f, 2.0f * random_float(state) - 1.0f, 2.0f * random_float(state) - 1.0f);
    float a = 2.0f * random_float(state) - 1.0f, b = 2.0f * random_float(state) - 1.0f;
    float l = sqrtf(p.squared_length() + a * a + b * b);
    p /= l;
    return p;
}

__device__ vec3 random_on_unit_sphere(curandState_t* state)
{
    vec3 p;
    while(true)
    {
        p.X = 2.0 * random_float(state) - 1.0f;
        p.Y = 2.0 * random_float(state) - 1.0f;
        p.Z = 2.0 * random_float(state) - 1.0f;
        if(dot(p, p) < 1.0f)
            return p.unit_vector();
    }
}

__device__ vec3 random_cosine_direction(curandState_t* state)
{
    float r1 = random_float(state);
    float r2 = random_float(state);
    float z = sqrtf(1.0f - r2);
    float phi = 2 * M_PI * r1;
    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);
    return vec3(x, y, z);
}