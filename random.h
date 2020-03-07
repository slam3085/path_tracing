#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include "vec3.h"

__device__ float random_float(curandState_t* state);
__device__ vec3 random_in_unit_sphere(curandState_t* state);
__device__ vec3 random_on_unit_sphere(curandState_t* state);
__device__ vec3 random_cosine_direction(curandState_t* state);