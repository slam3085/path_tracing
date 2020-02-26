#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "camera.h"

__device__ Camera::Camera()
{
    lower_left_corner = { -2.0, -1.0, -1.0 };
    horizontal = { 4.0, 0.0, 0.0 };
    vertical = { 0.0, 2.0, 0.0 };
    origin = { 0.0, 0.0, 0.0 };
}

__device__ Ray Camera::get_ray(float u, float v) const
{
    return {
        origin,
        lower_left_corner + horizontal * u + vertical * v
    };
}