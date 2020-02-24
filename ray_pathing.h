#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"

cudaError_t ray_pathing_with_cuda(vec3* framebuffer, int height, int width);