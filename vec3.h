#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct vec3 
{
    float X, Y, Z;
    __device__ float length() const;
    __device__ float squared_length() const;
    __device__ vec3 unit_vector() const;
    __device__ vec3 operator-() const;
    __device__ vec3& operator+=(const vec3 &v2);
    __device__ vec3& operator*=(float n);
    __device__ vec3& operator*=(const vec3 &v2);
    __device__ vec3& operator/=(float n);
};

__device__ vec3 operator+(const vec3& v1, const vec3& v2);
__device__ vec3 operator-(const vec3& v1, const vec3& v2);
__device__ vec3 operator*(const vec3& v1, const vec3& v2);
__device__ vec3 operator*(const vec3& v1, float n);
__device__ vec3 operator/(const vec3& v1, float n);
__device__ float dot(const vec3& v1, const vec3& v2);