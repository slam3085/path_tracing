#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t gradientWithCuda(float* framebuffer, int height, int width);