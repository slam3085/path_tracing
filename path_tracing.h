#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include "vec3.h"

void path_tracing_with_cuda(std::string filename, int height, int width);