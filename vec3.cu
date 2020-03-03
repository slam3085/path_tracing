#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "vec3.h"

__device__ float vec3::operator[](int i) const
{
    switch(i)
    {
        case 0:
            return X;
        case 1:
            return Y;
        case 2:
            return Z;
    }
}

__device__ float& vec3::operator[](int i)
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

__device__ float vec3::length() const
{
    return sqrt(X * X + Y * Y + Z * Z);
}

__device__ float vec3::squared_length() const
{
    return X * X + Y * Y + Z * Z;
}

__device__ vec3 vec3::unit_vector() const
{
    float l = length();
    return { X / l, Y / l, Z / l };
}

__device__ vec3 vec3::operator-() const
{
    return { -X, -Y, -Z };
}

__device__ vec3 operator+(const vec3& v1, const vec3& v2)
{
    return { v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z };
}

__device__ vec3 operator-(const vec3& v1, const vec3& v2)
{
    return { v1.X - v2.X, v1.Y - v2.Y, v1.Z - v2.Z };
}

__device__ vec3 operator*(const vec3& v1, const vec3& v2)
{
    return { v1.X * v2.X, v1.Y * v2.Y, v1.Z * v2.Z };
}

__device__ vec3 operator*(const vec3& v1, float n)
{
    return { v1.X * n, v1.Y * n, v1.Z * n };
}

__device__ vec3 operator/(const vec3& v1, float n)
{
    return { v1.X / n, v1.Y / n, v1.Z / n };
}

__device__ float dot(const vec3& v1, const vec3& v2)
{
    return v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z;
}

__device__ vec3& vec3::operator+=(const vec3 &v2)
{
    X += v2.X;
    Y += v2.Y;
    Z += v2.Z;
    return *this;
}

__device__ vec3& vec3::operator*=(float n)
{
    X *= n;
    Y *= n;
    Z *= n;
    return *this;
}

__device__ vec3& vec3::operator*=(const vec3 &v2)
{
    X *= v2.X;
    Y *= v2.Y;
    Z *= v2.Z;
    return *this;
}

__device__ vec3& vec3::operator/=(float n)
{
    X /= n;
    Y /= n;
    Z /= n;
    return *this;
}

__device__ vec3 cross(const vec3& v1, const vec3& v2)
{
    return {
        v1.Y * v2.Z - v1.Z * v2.Y,
        v1.Z * v2.X - v1.X * v2.Z,
        v1.X * v2.Y - v1.Y * v2.X
    };
}