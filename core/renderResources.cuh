#pragma once


#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <math_functions.h>
#include <device_functions.h>
#include <curand_kernel.h>
#include <cuda_texture_types.h>
#include <cuda_texture_types.h>

#include <vector_types.h>
#include <algorithm>

#include "vector.cuh"

extern texture<float, cudaTextureType2D, cudaReadModeElementType> _HGLut;


extern texture<float, cudaTextureType3D, cudaReadModeElementType> _DensityVolume;

#define Mip(i) _Mips##i
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips0;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips1;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips2;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips3;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips4;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips5;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips6;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips7;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips8;
#define TR_Mip(i) _TR_Mips##i
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips0;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips1;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips2;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips3;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips4;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips5;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips6;
extern texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips7;

extern texture<float4, cudaTextureType2D> _HDRI;

#define MipDensityStatic(mip, pos) tex3D<float>(Mip(mip), (pos).z + 0.5, (pos).y + 0.5, (pos).x + 0.5)
#define MipTrStatic(mip, pos) tex3D<float>(TR_Mip(mip), (pos).z + 0.5, (pos).y + 0.5, (pos).x + 0.5)

__device__ float MipDensityDynamic(int mip, float3 pos);
__device__ float MipTrDynamic(int mip, float3 pos);

//#define MipDensity MipDensityStatic
#define MipDensity MipDensityDynamic
#define MipTr MipTrDynamic


__device__ float3 ShadowTerm_TRTex(float3 ori, float3 lightDir, float3 dir, float3 lightColor, float g, int mip);