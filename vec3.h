#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct vec3 
{
    float X, Y, Z;
    __device__ __forceinline__ vec3() {}
    __device__ __forceinline__ vec3(float x, float y, float z): X(x), Y(y), Z(z) {}
    __device__ __forceinline__ float operator[](int i) const
    {
        switch (i)
        {
        case 0:
            return X;
        case 1:
            return Y;
        case 2:
            return Z;
        }
    }
    __device__ __forceinline__ float& operator[](int i)
    {
        switch (i)
        {
        case 0:
            return X;
        case 1:
            return Y;
        case 2:
            return Z;
        }
    }
    __device__ __forceinline__ float length() const
    {
        return sqrtf(X * X + Y * Y + Z * Z);
    }
    __device__ __forceinline__ float squared_length() const
    {
        return X * X + Y * Y + Z * Z;
    }
    __device__ __forceinline__ vec3 unit_vector() const
    {
        float l = length();
        return vec3(X / l, Y / l, Z / l);
    }
    __device__ __forceinline__ vec3 operator-() const
    {
        return vec3(-X, -Y, -Z);
    }
    __device__ __forceinline__ vec3& operator+=(const vec3 &v2)
    {
        X += v2.X;
        Y += v2.Y;
        Z += v2.Z;
        return *this;
    }
    __device__ __forceinline__ vec3& operator*=(float n)
    {
        X *= n;
        Y *= n;
        Z *= n;
        return *this;
    }
    __device__ __forceinline__ vec3& operator*=(const vec3 &v2)
    {
        X *= v2.X;
        Y *= v2.Y;
        Z *= v2.Z;
        return *this;
    }
    __device__ __forceinline__ vec3& operator/=(float n)
    {
        X /= n;
        Y /= n;
        Z /= n;
        return *this;
    }
};

__device__ __forceinline__ vec3 operator+(const vec3& v1, const vec3& v2)
{
    return vec3(v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z);
}

__device__ __forceinline__ vec3 operator-(const vec3& v1, const vec3& v2)
{
    return vec3(v1.X - v2.X, v1.Y - v2.Y, v1.Z - v2.Z);
}

__device__ __forceinline__ vec3 operator*(const vec3& v1, const vec3& v2)
{
    return vec3(v1.X * v2.X, v1.Y * v2.Y, v1.Z * v2.Z);
}

__device__ __forceinline__ vec3 operator*(const vec3& v1, float n)
{
    return vec3(v1.X * n, v1.Y * n, v1.Z * n);
}

__device__ __forceinline__ vec3 operator/(const vec3& v1, float n)
{
    return vec3(v1.X / n, v1.Y / n, v1.Z / n);
}

__device__ __forceinline__ float dot(const vec3& v1, const vec3& v2)
{
    return v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z;
}

__device__ __forceinline__ vec3 cross(const vec3& v1, const vec3& v2)
{
    return vec3(
    v1.Y * v2.Z - v1.Z * v2.Y,
    v1.Z * v2.X - v1.X * v2.Z,
    v1.X * v2.Y - v1.Y * v2.X
    );
}