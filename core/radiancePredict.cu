#include "radiancePredict.cuh"


#include "vector.cuh"
#include "tensor.cuh"


// ---------------------------------------------------------
// Operators
// ---------------------------------------------------------
__device__ inline void AVG(const float* In, float* Out,
    const int Length,
    const int ReadAt, const int WriteAt)
{
    Out[WriteAt] = 0.0;
    for (int i = 0; i < Length; i++)
        Out[WriteAt] += In[ReadAt + i];
    Out[WriteAt] /= float(Length);
}

__device__ inline void AVGMAX(const float* In, float* Out,
    const int Length,
    const int ReadAt, const int WriteAt, const int WriteAt2)
{
    Out[WriteAt] = 0.0;
    Out[WriteAt2] = -10000.0;
    for (int i = 0; i < Length; i++)
    {
        Out[WriteAt] += In[ReadAt + i];
        Out[WriteAt2] = max(Out[WriteAt2], In[ReadAt + i]);
    }
    Out[WriteAt] /= float(Length);
}

__device__ inline void MUL(float* In, const int Length, const int ReadAt, const float Mul)
{
    for (int i = 0; i < Length; i++)
        In[ReadAt + i] *= Mul;
};

__device__ inline void LinearReLU(const float* In, float* Out,
    const int From, const int To,
    const int ReadAt, const int WriteAt,
    const float* Weight, const float* Bias)
{
    for (int i = 0; i < To; i++)
    {
        Out[i + WriteAt] = 0.0;
        for (int j = 0; j < From; j++)
        {
            Out[i + WriteAt] += In[j + ReadAt] * Weight[j + i * From];
        }
        Out[i + WriteAt] += Bias[i];
        Out[i + WriteAt] = (Out[i + WriteAt] >= 0.0 ? Out[i + WriteAt] : 0.0);
    }
}

__device__ inline void LinearReLU_Unbias(const float* In, float* Out,
    const int From, const int To,
    const int ReadAt, const int WriteAt,
    const float* Weight)
{
    for (int i = 0; i < To; i++)
    {
        Out[i + WriteAt] = 0.0;
        for (int j = 0; j < From; j++)
        {
            Out[i + WriteAt] += In[j + ReadAt] * Weight[j + i * From];
        }
        Out[i + WriteAt] = (Out[i + WriteAt] >= 0.0 ? Out[i + WriteAt] : 0.0);
    }
}

__device__ inline void LinearSm_Unbias(const float* In, float* Out,
    const int From, const int To,
    const int ReadAt, const int WriteAt,
    const float* Weight)
{
    for (int i = 0; i < To; i++)
    {
        Out[i + WriteAt] = 0.0;
        for (int j = 0; j < From; j++)
            Out[i + WriteAt] += In[j + ReadAt] * Weight[j + i * From];
        Out[i + WriteAt] = 1.0 / (1.0 + exp(-Out[i + WriteAt]));
    }
};

__device__ inline void Linear(const float* In, float* Out,
    const int From, const int To,
    const int ReadAt, const int WriteAt,
    const float* Weight, const float* Bias)
{
    for (int i = 0; i < To; i++)
    {
        Out[i + WriteAt] = 0.0;
        for (int j = 0; j < From; j++)
        {
            Out[i + WriteAt] += In[j + ReadAt] * Weight[j + i * From];
        }
        Out[i + WriteAt] += Bias[i];
    }
}

__device__ inline void Linear_Unbias(const float* In, float* Out,
    const int From, const int To,
    const int ReadAt, const int WriteAt,
    const float* Weight)
{
    for (int i = 0; i < To; i++)
    {
        Out[i + WriteAt] = 0.0;
        for (int j = 0; j < From; j++)
        {
            Out[i + WriteAt] += In[j + ReadAt] * Weight[j + i * From];
        }
    }
}

__device__ inline void ReLU(float* In, const int Length, const int ReadAt)
{
    for (int i = 0; i < Length; i++)
    {
        In[ReadAt + i] = In[ReadAt + i] >= 0.0 ? In[ReadAt + i] : 0.0;
    }
};

#pragma region LinearRelue

template<int From, int To>
__device__ inline void LinearReLU_SFL
(const float* In, float* Out,
    const float* Weight, const float* Bias,
    const int ReadAt = 0, const int WriteAt = 0)
{
#ifdef ENBALE_TENSOR
    if constexpr ((From % 16 == 0) && (To % 16 == 0)) {
        WMMA_Relu<To, From>(Weight, In + ReadAt, Bias, Out + WriteAt);
        return;
    }
#endif
    static_assert((From % 32 == 0), "From must be integral multiple of 32.");
    int lane_id = __lane_id();
    for (int i = 0; i < To; i++)
    {
        Out[i + WriteAt] = Bias[i];
        float weight = Weight[lane_id + i * From];
        for (int k = 0; k < From / 32 - 1; k++)
        {
            float weight2 = Weight[lane_id + i * From + (k + 1) * 32];
            for (int j = 0; j < 32; j++)
                Out[i + WriteAt] += In[j + k * 32 + ReadAt] * __shfl_sync(0xFFFFFFFFU, weight, j);
            weight = weight2;
        }
        for (int j = 0; j < 32; j++)
            Out[i + WriteAt] += In[j + (From - 32) + ReadAt] * __shfl_sync(0xFFFFFFFFU, weight, j);
        Out[i + WriteAt] = (Out[i + WriteAt] >= 0.0 ? Out[i + WriteAt] : 0.0);
    }
}
template<int Size>
__device__ inline void LinearReLU
(const float* In, float* Out,
    const float* Weight, const float* Bias,
    const int ReadAt = 0, const int WriteAt = 0)
{
    static_assert((Size < 32), "Size must be less than 32.");
#ifdef ENBALE_TENSOR
    if constexpr (Size == 16) {
        WMMA_Relu<Size, Size>(Weight, In + ReadAt, Bias, Out + WriteAt);
        return;
    }
#endif
    for (int i = 0; i < Size; i++)
    {
        Out[i + WriteAt] = Bias[i];
        for (int j = 0; j < Size; j++)
        {
            Out[i + WriteAt] += In[j + ReadAt] * Weight[j + i * Size];
        }
        Out[i + WriteAt] = (Out[i + WriteAt] >= 0.0 ? Out[i + WriteAt] : 0.0);
    }
}

template<int Size>
__device__ inline void LinearReLU_SFL
(const float* In, float* Out,
    const float* Weight, const float* Bias,
    const int ReadAt = 0, const int WriteAt = 0)
{
    if constexpr (Size < 32) {
        LinearReLU<Size>(In, Out, Weight, Bias, ReadAt, WriteAt);
    }
    else {
        LinearReLU_SFL<Size, Size>(In, Out, Weight, Bias, ReadAt, WriteAt);
    }
}
template<>
__device__ inline void LinearReLU_SFL<32>
(const float* In, float* Out,
    const float* Weight, const float* Bias,
    const int ReadAt, const int WriteAt)
{
#ifdef ENBALE_TENSOR
    WMMA_Relu<32, 32>(Weight, In + ReadAt, Bias, Out + WriteAt);
    return;
#endif
    int lane_id = __lane_id();
    float weight = Weight[lane_id];
    for (int i = 0; i < 31; i++)
    {
        float weight2 = Weight[lane_id + (i + 1) * 32];
        Out[i + WriteAt] = Bias[i];
        for (int j = 0; j < 32; j++)
            Out[i + WriteAt] += In[j + ReadAt] * __shfl_sync(0xFFFFFFFFU, weight, j);
        Out[i + WriteAt] = (Out[i + WriteAt] >= 0.0 ? Out[i + WriteAt] : 0.0);
        weight = weight2;
    }
    {
        int i = 31;
        Out[i + WriteAt] = Bias[i];
        for (int j = 0; j < 32; j++)
            Out[i + WriteAt] += In[j + ReadAt] * __shfl_sync(0xFFFFFFFFU, weight, j);
        Out[i + WriteAt] = (Out[i + WriteAt] >= 0.0 ? Out[i + WriteAt] : 0.0);
    }
}

#pragma endregion

__device__ inline void Add(float* A, const float* B, const int Size, const int A_At, const int B_At)
{
    for (int i = 0; i < Size; i++)
    {
        A[i + A_At] = A[i + A_At] + B[i + B_At];
    }
}
__device__ inline void Rep(float* A, const float* B, const int Size, const int A_At, const int B_At)
{
    memcpy(A + A_At, B + B_At, Size * sizeof(float));
}

#define SE(FROM,TO,SIZEX,PID,SEW0,SEW1,WW,TRW,HGW) {\
    AVGMAX(X_Val    , AvgPool, SIZEX, FROM, PID + 0, POOL + PID + 0);\
    AVGMAX(X_Val_Sub, AvgPool, SIZEX, FROM, PID + 1, POOL + PID + 1);\
    AVGMAX(X_Val_Hg , AvgPool, SIZEX, FROM, PID + 2, POOL + PID + 2);\
    Rep(TempAvgPool, AvgPool, 3, 0, PID);\
    Rep(TempAvgPool, AvgPool, 3, 3, POOL + PID);\
    LinearReLU_Unbias(TempAvgPool, AvgWeight, 8, 8, 0, 0, SEW0);\
    LinearSm_Unbias(AvgWeight, AvgWeightPool, 8, 3, 0, 0, SEW1);\
    Rep(X_ValA, X_Val, SIZEX, TO, FROM);\
    MUL(X_ValA, SIZEX, TO, AvgWeightPool[0]);\
    LinearReLU_SFL<SIZEX>(X_ValA, X_ValA2, WW##W##, WW##B##,TO,TO);\
    Rep(X_ValB, X_Val_Sub, SIZEX, TO, FROM);\
    MUL(X_ValB, SIZEX, TO, AvgWeightPool[1]);\
    LinearReLU_SFL<SIZEX>(X_ValB, X_ValB2, TRW##W##, TRW##B##,TO,TO);\
    Rep(X_ValC, X_Val_Hg, SIZEX, TO, FROM);\
    MUL(X_ValC, SIZEX, TO, AvgWeightPool[2]);\
    LinearReLU_SFL<SIZEX>(X_ValC, X_ValC2, HGW##W##, HGW##B##,TO,TO);\
    }\

#define SERES(LAST,FROM,TO,SIZEX,PID,SEW0,SEW1,WW,TRW,HGW) {\
    AVGMAX(X_Val    , AvgPool, SIZEX, FROM, PID + 0, POOL + PID + 0);\
    AVGMAX(X_Val_Sub, AvgPool, SIZEX, FROM, PID + 1, POOL + PID + 1);\
    AVGMAX(X_Val_Hg , AvgPool, SIZEX, FROM, PID + 2, POOL + PID + 2);\
    Rep(TempAvgPool, AvgPool, 3, 0, PID);\
    Rep(TempAvgPool, AvgPool, 3, 3, POOL + PID);\
    LinearReLU_Unbias(TempAvgPool, AvgWeight, 8, 8, 0, 0, SEW0);\
    LinearSm_Unbias(AvgWeight, AvgWeightPool, 8, 3, 0, 0, SEW1);\
    Rep(X_ValA, X_Val, SIZEX, TO, FROM);\
    MUL(X_ValA, SIZEX, TO, AvgWeightPool[0]);\
    Add(X_ValA, X_ValA2, SIZEX, TO, LAST);\
    LinearReLU_SFL<SIZEX>(X_ValA, X_ValA2, WW##W##, WW##B##,TO,TO);\
    Rep(X_ValB, X_Val_Sub, SIZEX, TO, FROM);\
    MUL(X_ValB, SIZEX, TO, AvgWeightPool[1]);\
    Add(X_ValB, X_ValB2, SIZEX, TO, LAST);\
    LinearReLU_SFL<SIZEX>(X_ValB, X_ValB2, TRW##W##, TRW##B##,TO,TO);\
    Rep(X_ValC, X_Val_Hg, SIZEX, TO, FROM);\
    MUL(X_ValC, SIZEX, TO, AvgWeightPool[2]);\
    Add(X_ValC, X_ValC2, SIZEX, TO, LAST);\
    LinearReLU_SFL<SIZEX>(X_ValC, X_ValC2, HGW##W##, HGW##B##,TO,TO);\
    }\

#define ADD3(FROM,TO,SIZEX){\
    Add(X_ValA2, X_Val, SIZEX, TO, FROM);\
    Add(X_ValB2, X_Val_Sub, SIZEX, TO, FROM);\
    Add(X_ValC2, X_Val_Hg, SIZEX, TO, FROM);\
    }\

#define REP3(FROM,TO,SIZEX){\
    Rep(X_ValA2, X_ValA2, SIZEX, FROM, TO);\
    Rep(X_ValA2, X_Val, SIZEX, TO, TO);\
    Rep(X_ValB2, X_ValB2, SIZEX, FROM, TO);\
    Rep(X_ValB2, X_Val_Sub, SIZEX, TO, TO);\
    Rep(X_ValC2, X_ValC2, SIZEX, FROM, TO);\
    Rep(X_ValC2, X_Val_Hg, SIZEX, TO, TO);\
    }\







// ---------------------------------------------------------
// Networks
// ---------------------------------------------------------
#include "NNWeight.cuh"

__device__ float3 RadiancePredict(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int PC = 192;
    const int DI = 160;
    const int POOL = 36;
    const int P_DI = 24;

    float X_Val[PC];
    float X_Val_Sub[PC];
    float X_Val_Hg[PC];
    int randIndex = 0;

    if (active) {
        // perform sample
        for (int i = 0; i < PC; i++)
        {
            Offset_Layer_ CurrentOffsetInfo = GetSamples23_(i);
            CurrentOffsetInfo.Layer += 0.1f;
            float3 CurrentOffsetPos;
            if (CurrentOffsetInfo.type >= 4)
            {
                CurrentOffsetPos = pos + SphereRandom3(randIndex, CurrentOffsetInfo.Offset, XMain, YMain, ZMain, g);
                randIndex++;
            }
            else
            {
                CurrentOffsetPos = pos + LightDir * CurrentOffsetInfo.Offset;
            }
            X_Val[i] = log(1.0 + alpha * MipDensityDynamic(int(CurrentOffsetInfo.Layer), CurrentOffsetPos) / 64.0f);

            float3 MsDir;
            if (CurrentOffsetInfo.localindex == 0)
            {
                MsDir = XMain;
                X_Val_Hg[i] = log(HenyeyGreenstein(dot(MsDir, LightDir), g) + 1.0f);
            }
            else
            {
                MsDir = normalize(CurrentOffsetPos - pos);
                float Radius = float(1 << int(CurrentOffsetInfo.Layer)) / 256.0;
                float Angle = atan(0.5f * Radius / CurrentOffsetInfo.Offset);
                float cos = dot(XMain, MsDir);
                float cos2 = dot(LXMain, MsDir);
                float u = cos * 0.5 + 0.5;
                float u2 = cos2 * 0.5 + 0.5;
                float v = Angle / (3.1415926535 * 60.0 / 180.0);
                float HG0 = tex2D<float>(_HGLut, u, v);
                float HG1 = tex2D<float>(_HGLut, u2, v);
                X_Val_Hg[i] = log(HG0 * HG1 + 1.0f);
            }
            //X_Val_Sub[i] = TR_MipDensityDynamic(max(int(CurrentOffsetInfo.Layer) - 1, 0), CurrentOffsetPos);
            X_Val_Sub[i] = ShadowTerm_TRTex(CurrentOffsetPos, LXMain, XMain, float3{ 1.0f,1.0f,1.0f }, g, max(int(CurrentOffsetInfo.Layer) - 1, 0)).x;
        }
    }
    else {
        // fill with fake data
        for (int i = 0; i < PC; i++) {
            X_Val[i] = 0;
            X_Val_Hg[i] = 0;
            X_Val_Sub[i] = 0;
        }
    }

    float X_ValA[160];
    float X_ValB[160];
    float X_ValC[160];
    float X_ValA2[160];
    float X_ValB2[160];
    float X_ValC2[160];
    float AvgPool[75];//Global
    float AvgWeight[16];//Local
    float AvgWeightPool[6];//Local
    float TempAvgPool[8];
    float Comb[128];

    float scbasex[1];
    float scbasey[1];
    float scbasez[1];
    float gamma = acos(dot(XMain, LXMain));
    TempAvgPool[6] = g;
    TempAvgPool[7] = gamma;
    float3 srp = pow(scatterrate, 4.0f);
    scbasex[0] = srp.x;
    scbasey[0] = srp.y;
    scbasez[0] = srp.z;
    AvgPool[POOL + POOL] = g;
    AvgPool[POOL + POOL + 1] = gamma;
    __syncwarp();
    // run NN
    SE(0, 0, 8, 0, LSE01W, LSE02W, L01, L_Tr01, L_Hg01)
        ADD3(0, 0, 8)
        //
        SERES(0, 8, 8, 8, 3, LSE11W, LSE12W, L11, L_Tr11, L_Hg11)
        REP3(0, 8, 8)
        //
        SERES(0, 16, 16, 16, 6, LSE21W, LSE22W, L21, L_Tr21, L_Hg21)
        ADD3(16, 16, 16)
        //
        SERES(16, 32, 32, 16, 9, LSE31W, LSE32W, L31, L_Tr31, L_Hg31)
        ADD3(32, 32, 16)
        //
        SERES(32, 48, 48, 16, 12, LSE41W, LSE42W, L41, L_Tr41, L_Hg41)
        REP3(32, 48, 16)
        //
        SERES(32, 64, 64, 32, 15, LSE51W, LSE52W, L51, L_Tr51, L_Hg51)
        ADD3(64, 64, 32)
        //
        SERES(64, 96, 96, 32, 18, LSE61W, LSE62W, L61, L_Tr61, L_Hg61)
        ADD3(96, 96, 32)
        //Additional
        SERES(96, 128, 128, 32, 21, LSE71W, LSE72W, L71, L_Tr71, L_Hg71)
        ADD3(128, 128, 32)
        LinearReLU_SFL<32>(X_ValA2, Comb, L81W, L81B, 128, 0);//Last layer
    LinearReLU_SFL<32>(X_ValB2, Comb, L_Tr81W, L_Tr81B, 128, 32);
    LinearReLU_SFL<32>(X_ValC2, Comb, L_Hg81W, L_Hg81B, 128, 64);
    //
    //di
    SE(DI + 0, 0, 8, P_DI + 0, LDSE01W, LDSE02W, LD01, LD_Tr01, LD_Hg01)
        ADD3(DI + 0, 0, 8)
        //di1
        SERES(0, DI + 8, 8, 8, P_DI + 3, LDSE11W, LDSE12W, LD11, LD_Tr11, LD_Hg11)
        ADD3(DI + 8, 8, 8)
        //di2
        SERES(8, DI + 16, 16, 8, P_DI + 6, LDSE21W, LDSE22W, LD21, LD_Tr21, LD_Hg21)
        ADD3(DI + 16, 16, 8)
        //di3
        SERES(16, DI + 24, 24, 8, P_DI + 9, LDSE31W, LDSE32W, LD31, LD_Tr31, LD_Hg31)
        ADD3(DI + 24, 24, 8)
        LinearReLU_SFL<8>(X_ValA2, Comb, LD41W, LD41B, 24, 96);//Last layer.parse in 0~8
    LinearReLU_SFL<8>(X_ValB2, Comb, LD_Tr41W, LD_Tr41B, 24, 104);
    LinearReLU_SFL<8>(X_ValC2, Comb, LD_Hg41W, LD_Hg41B, 24, 112);
    //
#define SC(p,P) { AvgPool[POOL + POOL + 2] = scbase##p##[0];\
    Rep(X_ValA, Comb, 120, 0, 0);\
    LinearReLU(AvgPool, X_ValA, 3, 8, POOL + POOL, 120, LGGSW, LGGSB);\
    LinearReLU_Unbias(AvgPool, AvgWeight, 75, 16, 0, 0, LSEFin1W);\
    LinearSm_Unbias(AvgWeight, AvgWeightPool, 16, 6, 0, 0, LSEFin2W);\
    MUL(X_ValA, 32, 0, AvgWeightPool[0]);\
    MUL(X_ValA, 32, 32, AvgWeightPool[1]);\
    MUL(X_ValA, 32, 64, AvgWeightPool[2]);\
    MUL(X_ValA, 8, 96, AvgWeightPool[3]);\
    MUL(X_ValA, 8, 104, AvgWeightPool[4]);\
    MUL(X_ValA, 8, 112, AvgWeightPool[5]);\
    LinearReLU_SFL<128, 128>(X_ValA, X_ValB, LC0W, LC0B);\
    LinearReLU_SFL<128, 64>(X_ValB, X_ValA, LC1W, LC1B);\
    LinearReLU_SFL<64, 32>(X_ValA, X_ValB, LC2W, LC2B);\
    LinearReLU_SFL<32, 16>(X_ValB, X_ValA, LXW, LXB);\
    LinearReLU_SFL<16>(X_ValA, X_ValB, LX0W, LX0B);\
    LinearReLU_SFL<16>(X_ValB, X_ValC, LX1W, LX1B);\
    Add(X_ValA, X_ValC, 16, 0, 0);\
    LinearReLU_SFL<16>(X_ValA, X_ValB, LX2W, LX2B);\
    LinearReLU_SFL<16>(X_ValB, X_ValC, LX3W, LX3B);\
    Add(X_ValA, X_ValC, 16, 0, 0);\
    LinearReLU_SFL<16>(X_ValA, X_ValB, LX4W, LX4B);\
    LinearReLU_SFL<16>(X_ValB, X_ValC, LX5W, LX5B);\
    Add(X_ValA, X_ValC, 16, 0, 0);\
    LinearReLU_SFL<16>(X_ValA, X_ValB, LX6W, LX6B);\
    LinearReLU_SFL<16>(X_ValB, X_ValC, LX7W, LX7B);\
    Add(X_ValA, X_ValC, 16, 0, 0);\
    LinearReLU_SFL<16>(X_ValA, X_ValB, LX8W, LX8B);\
    LinearReLU_SFL<16>(X_ValB, X_ValC, LX9W, LX9B);\
    Add(X_ValA, X_ValC, 16, 0, 0);\
    LinearReLU_SFL<16>(X_ValA, X_ValB, LX10W, LX10B);\
    LinearReLU_SFL<16>(X_ValB, X_ValC, LX11W, LX11B);\
    Add(X_ValA, X_ValC, 16, 0, 0);\
    LinearReLU(X_ValA, X_ValB, 16, 1, 0, 0, LX12W, LX12B);\
    P = X_ValB[0]; }\

    float X, Y, Z;
    SC(x, X);

    if (scatterrate.y == scatterrate.x) Y = X;
    else SC(y, Y);

    if (scatterrate.z == scatterrate.x) Z = X;
    else if (scatterrate.z == scatterrate.y) Z = Y;
    else  SC(z, Z);

    return max((exp(float3{ X,Y,Z })) - 1.0, float3{ 0 }) * srp;
}



#ifdef CRPNN
#include "NNWeight_RPNN.cuh"
__device__ float3 RadiancePredict_RPNN(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    const int L = 10;
    const int S = 225;
    const int N = 226;
    const int B = 200;
    float X_Val[2250];
    int randIndex = 0;

    float Gamma = acos(dot(XMain, LXMain));
    if (active)
    {
        const float3 eX = normalize(cross(LXMain, XMain));
        const float3 eY = normalize(cross(eX, LXMain));

        const float descSizeAtLevel0 = 0.5f / 1024.0f;
        float scale = 0.5f * descSizeAtLevel0;

        float mipmapLevel = 0;
        for (int i = 0; i < L; i++)
        {
            const int start_index = i * S;
            int local_index = 0;
            float currentmipmapLevel = max(min(mipmapLevel - 2.0f, 9.0f), 0.0f);
            for (int z = -2; z <= 6; z++)
            {
                for (int y = -2; y <= 2; y++)
                {
                    for (int x = -2; x <= 2; x++)
                    {
                        float3 offset = (eX * x + eY * y + LXMain * z) * scale;
                        const float3 samplePos = pos + offset;
                        float density = MipDensityDynamic(int(currentmipmapLevel + 0.001), samplePos);

                        X_Val[start_index + local_index] = density * alpha / 64.0f;
                        local_index++;
                    }
                }
            }
            scale *= 2.0;
            mipmapLevel += 1.0;
        }
    }
    else
    {
        // fill with fake data
        for (int i = 0; i < L * (S + 1); i++)
        {
            X_Val[i] = 0;
        }
    }

    float X_ValA[N];
    float X_ValB[N];
    float X_ValC[N];
    float X_ValC2[N];

    X_ValA[S] = Gamma;
    Rep(X_ValA, X_Val, S, 0, S * 0);//load
    Linear_Unbias(X_ValA, X_ValB, N, B, 0, 0, V0W);
    ReLU(X_ValB, B, 0);//F0
    Linear(X_ValB, X_ValC, B, B, 0, 0, U0W, U0B);//FF0
    ReLU(X_ValC, B, 0);//o0

    Rep(X_ValA, X_Val, S, 0, S * 1);
    Linear_Unbias(X_ValA, X_ValB, N, B, 0, 0, V1W);
    Linear(X_ValC, X_ValC2, B, B, 0, 0, W1W, W1B);
    Add(X_ValB, X_ValC2, B, 0, 0);
    ReLU(X_ValB, B, 0);//F1
    Linear(X_ValB, X_ValC2, B, B, 0, 0, U1W, U1B);//FF1
    Add(X_ValC, X_ValC2, B, 0, 0);
    ReLU(X_ValC, B, 0);//o1

    Rep(X_ValA, X_Val, S, 0, S * 2);
    Linear_Unbias(X_ValA, X_ValB, N, B, 0, 0, V2W);
    Linear(X_ValC, X_ValC2, B, B, 0, 0, W2W, W2B);
    Add(X_ValB, X_ValC2, B, 0, 0);
    ReLU(X_ValB, B, 0);//F2
    Linear(X_ValB, X_ValC2, B, B, 0, 0, U2W, U2B);//FF2
    Add(X_ValC, X_ValC2, B, 0, 0);
    ReLU(X_ValC, B, 0);//o2

    Rep(X_ValA, X_Val, S, 0, S * 3);
    Linear_Unbias(X_ValA, X_ValB, N, B, 0, 0, V3W);
    Linear(X_ValC, X_ValC2, B, B, 0, 0, W3W, W3B);
    Add(X_ValB, X_ValC2, B, 0, 0);
    ReLU(X_ValB, B, 0);//F3
    Linear(X_ValB, X_ValC2, B, B, 0, 0, U3W, U3B);//FF3
    Add(X_ValC, X_ValC2, B, 0, 0);
    ReLU(X_ValC, B, 0);//o3


    Rep(X_ValA, X_Val, S, 0, S * 4);
    Linear_Unbias(X_ValA, X_ValB, N, B, 0, 0, V4W);
    Linear(X_ValC, X_ValC2, B, B, 0, 0, W4W, W4B);
    Add(X_ValB, X_ValC2, B, 0, 0);
    ReLU(X_ValB, B, 0);
    Linear(X_ValB, X_ValC2, B, B, 0, 0, U4W, U4B);
    Add(X_ValC, X_ValC2, B, 0, 0);
    ReLU(X_ValC, B, 0);

    Rep(X_ValA, X_Val, S, 0, S * 5);
    Linear_Unbias(X_ValA, X_ValB, N, B, 0, 0, V5W);
    Linear(X_ValC, X_ValC2, B, B, 0, 0, W5W, W5B);
    Add(X_ValB, X_ValC2, B, 0, 0);
    ReLU(X_ValB, B, 0);
    Linear(X_ValB, X_ValC2, B, B, 0, 0, U5W, U5B);
    Add(X_ValC, X_ValC2, B, 0, 0);
    ReLU(X_ValC, B, 0);

    Rep(X_ValA, X_Val, S, 0, S * 6);
    Linear_Unbias(X_ValA, X_ValB, N, B, 0, 0, V6W);
    Linear(X_ValC, X_ValC2, B, B, 0, 0, W6W, W6B);
    Add(X_ValB, X_ValC2, B, 0, 0);
    ReLU(X_ValB, B, 0);
    Linear(X_ValB, X_ValC2, B, B, 0, 0, U6W, U6B);
    Add(X_ValC, X_ValC2, B, 0, 0);
    ReLU(X_ValC, B, 0);

    Rep(X_ValA, X_Val, S, 0, S * 7);
    Linear_Unbias(X_ValA, X_ValB, N, B, 0, 0, V7W);
    Linear(X_ValC, X_ValC2, B, B, 0, 0, W7W, W7B);
    Add(X_ValB, X_ValC2, B, 0, 0);
    ReLU(X_ValB, B, 0);
    Linear(X_ValB, X_ValC2, B, B, 0, 0, U7W, U7B);
    Add(X_ValC, X_ValC2, B, 0, 0);
    ReLU(X_ValC, B, 0);

    Rep(X_ValA, X_Val, S, 0, S * 8);
    Linear_Unbias(X_ValA, X_ValB, N, B, 0, 0, V8W);
    Linear(X_ValC, X_ValC2, B, B, 0, 0, W8W, W8B);
    Add(X_ValB, X_ValC2, B, 0, 0);
    ReLU(X_ValB, B, 0);
    Linear(X_ValB, X_ValC2, B, B, 0, 0, U8W, U8B);
    Add(X_ValC, X_ValC2, B, 0, 0);
    ReLU(X_ValC, B, 0);

    Rep(X_ValA, X_Val, S, 0, S * 9);
    Linear_Unbias(X_ValA, X_ValB, N, B, 0, 0, V9W);
    Linear(X_ValC, X_ValC2, B, B, 0, 0, W9W, W9B);
    Add(X_ValB, X_ValC2, B, 0, 0);
    ReLU(X_ValB, B, 0);
    Linear(X_ValB, X_ValC2, B, B, 0, 0, U9W, U9B);
    Add(X_ValC, X_ValC2, B, 0, 0);
    ReLU(X_ValC, B, 0);

    LinearReLU(X_ValC, X_ValC2, B, B, 0, 0, F0W, F0B);
    LinearReLU(X_ValC2, X_ValC, B, B, 0, 0, F1W, F1B);
    LinearReLU(X_ValC, X_ValC2, B, 1, 0, 0, F2W, F2B);

    float output = X_ValC2[0];
    output = max(exp(output) - 1.0f, 0.0f);
    return float3{ output,output,output };
}
#else
__device__ float3 RadiancePredict_RPNN(curandState* seed, bool active, float3 pos, float3 LightDir, float3 XMain, float3 YMain, float3 ZMain, float3 LXMain, float3 LYMain, float3 LZMain, float alpha, float g, float3 scatterrate)
{
    return make_float3(0, 0, 0);
}
#endif