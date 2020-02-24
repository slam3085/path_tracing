#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"

cudaError_t gradientWithCuda(vec3* framebuffer, int height, int width);