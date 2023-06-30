#include "renderResources.cuh"


texture<float, cudaTextureType2D, cudaReadModeElementType> _HGLut;


texture<float, cudaTextureType3D, cudaReadModeElementType> _DensityVolume;

texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips0;
texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips1;
texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips2;
texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips3;
texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips4;
texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips5;
texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips6;
texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips7;
texture<float, cudaTextureType3D, cudaReadModeElementType> _Mips8;

texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips0;
texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips1;
texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips2;
texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips3;
texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips4;
texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips5;
texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips6;
texture<float, cudaTextureType3D, cudaReadModeElementType> _TR_Mips7;

texture<float4, cudaTextureType2D> _HDRI;


__device__ float MipDensityDynamic(int mip, float3 pos) {
    float3 uv = pos + 0.5;
    switch (mip)
    {
    case 0:
        return tex3D<float>(_Mips0, uv.z, uv.y, uv.x);
    case 1:
        return tex3D<float>(_Mips1, uv.z, uv.y, uv.x);
    case 2:
        return tex3D<float>(_Mips2, uv.z, uv.y, uv.x);
    case 3:
        return tex3D<float>(_Mips3, uv.z, uv.y, uv.x);
    case 4:
        return tex3D<float>(_Mips4, uv.z, uv.y, uv.x);
    case 5:
        return tex3D<float>(_Mips5, uv.z, uv.y, uv.x);
    case 6:
        return tex3D<float>(_Mips6, uv.z, uv.y, uv.x);
    case 7:
        return tex3D<float>(_Mips7, uv.z, uv.y, uv.x);
    case 8:
        return tex3D<float>(_Mips8, uv.z, uv.y, uv.x);
    default:
        return 0;
    }
}

__device__ float MipTrDynamic(int mip, float3 pos) {
    float3 uv = pos + 0.5;
    switch (mip)
    {
    case 0:
        return tex3D<float>(_TR_Mips0, uv.z, uv.y, uv.x);
    case 1:
        return tex3D<float>(_TR_Mips1, uv.z, uv.y, uv.x);
    case 2:
        return tex3D<float>(_TR_Mips2, uv.z, uv.y, uv.x);
    case 3:
        return tex3D<float>(_TR_Mips3, uv.z, uv.y, uv.x);
    case 4:
        return tex3D<float>(_TR_Mips4, uv.z, uv.y, uv.x);
    case 5:
        return tex3D<float>(_TR_Mips5, uv.z, uv.y, uv.x);
    case 6:
        return tex3D<float>(_TR_Mips6, uv.z, uv.y, uv.x);
    case 7:
        return tex3D<float>(_TR_Mips7, uv.z, uv.y, uv.x);
    default:
        return 0;
    }
}

__device__ float3 ShadowTerm_TRTex(float3 ori, float3 lightDir, float3 dir, float3 lightColor, float g, int mip)
{
    if (ori.x < -0.5 || ori.y < -0.5 || ori.z < -0.5 || ori.x > 0.5 || ori.y > 0.5 || ori.z > 0.5)
    {
        float offset = RayBoxOffset(ori, lightDir);
        if (offset >= 0)
        {
            return lightColor * MipTr(mip, ori + lightDir * offset);
        }
        else
        {
            return lightColor * float3{ 1.0f,1.0f,1.0f };
        }
    }
    else
    {
        return lightColor * MipTr(mip, ori);
    }
}