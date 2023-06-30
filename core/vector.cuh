#pragma once

#include <vector_types.h>
#include <algorithm>

__device__ __host__ const float3 operator*(const float3 a, const float b);
__device__ __host__ const float3 operator*(const float3 a, const float3 b);

__device__ __host__ const float3 operator/(const float3 a, const float b);
__device__ __host__ const float3 operator/(const float3 a, const float3 b);

__device__ __host__ const float3 operator+(const float3 a, const float b);
__device__ __host__ const float3 operator+(const float3 a, const float3 b);

__device__ __host__ const float3 operator-(const float3 a, const float b);
__device__ __host__ const float3 operator-(const float3 a, const float3 b);
__device__ __host__ const float3 operator-(const float3 a);

__device__ __host__ const float4 make_float4(const float3 a, const float b = 0);
__device__ __host__ const float3 make_float3(const float4 a);
__device__ __host__ const float3 inv(const float3 a);
__device__ __host__ const float frac(const float f);
__device__ __host__ const float3 frac(const float3 f);
__device__ __host__ const float dot(const float3 a, const float3 b);
__device__ __host__ const float saturate_(const float v);
__device__ __host__ const float3 saturate_(const float3 v);
__device__ const float3 saturate(const float3 v);
__device__ __host__ const float3 max(const float3 a, const float3 b);
__device__ __host__ const float3 min(const float3 a, const float3 b);
__device__ __host__ const float3 normalize(const float3 v);
__device__ __host__ const float lerp(const float a, const float b, const float v);
__device__ __host__ const float3 lerp(const float3 a, const float3 b, const float v);
__device__ __host__ const float3 cross(const float3 a, const float3 b);
__device__ __host__ const float3 pow(const float3 a, const float b); 
__device__ __host__ const float3 exp(const float3 a);
__device__ __host__ const float sign(const float a);
__device__ __host__ const float3 sign(const float3 a);
__device__ __host__ const float3 abs(const float3 a);
__device__ __host__ const float3 sin(const float3 a);
__device__ __host__ const int3 floor(const float3 a);
__device__ __host__ const float length(const float3 a);
__device__ __host__ const float distance(const float3 a, const float3 b);
__device__ __host__ float RayBoxOffset(float3 p, float3 dir);
__device__ __host__ float RayBoxDistance(float3 p, float3 dir);

__device__ __host__ const float3 Roberts2(const int n);
__device__ __host__ const float3 UniformSampleHemisphere(const float x, const float y);

__device__ __host__ float3 UniformSampleSphere(float2 E);

__device__ __host__ float HenyeyGreenstein(float cos, float g);
__device__ __host__ float HenyeyGreenstein_Avg(float3 ViewDir,float3 Dir, float Offset, float Radius, float g);
__device__ __host__ float SampleHeneyGreenstein(float s, float g);
__device__ __host__ float3 SampleHenyeyGreenstein(const float e0, const float e1, const float3 v, const float g);
__device__ __host__ float3 SampleHenyeyGreenstein_HG(const float e0, const float e1, const float3 v, const float g,float& hg);
__device__ __host__ float3 SampleHenyeyGreenstein(const int index, const float3 XMain, const float3 YMain, const float3 ZMain, const float g);

__device__ __host__ struct float3x3 {
    float3 x, y, z;
    __device__ __host__ float3x3(float3 x = {0}, float3 y = {0}, float3 z = {0}) : x(x), y(y), z(z) {}
    __device__ __host__ const float3 operator*(const float3 v) const;
};

__device__ __host__ const float3 SphereRandom3(int index, float radius, float3 XMain, float3 YMain, float3 ZMain,float g);

__device__ __host__ const float3 HGRandom(int index, int& additionalMip, float3 XMain, float3 YMain, float3 ZMain,float g);

__device__ __host__ struct Offset_Layer_
{
	float Offset = 0.0;
	float Layer = 0.0;
	int type = 0;
	int localindex = 0;
	int subindex = 0;
	Offset_Layer_(float noff, float nl, int type_ = 0,int localindex_ = 0,int subindex_ = 0)
	{
		Offset = noff; Layer = nl; type = type_; localindex = localindex_; subindex = subindex_;
	}
};
__device__ __host__ const Offset_Layer_ GetSamples23_(int index);
__device__ __host__ const Offset_Layer_ GetSamples24_(int index);