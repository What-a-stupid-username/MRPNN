#include "volume.hpp"

#include "hdr_loader.h"

#include "render.cuh"

#include "platform.h"
#include <thread>

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#define CheckError { auto error = cudaGetLastError(); if (error != 0) cout << cudaGetErrorString(error); }

template<int type>
__global__ void CalculateRadianceMulti(volatile int* record, float3* result, float3* ori, float3* dir, float3 lightDir, float3 lightColor = { 1, 1, 1 }, float alpha = 1, int multiScatter = 1, float g = 0, int sampleNum = 1) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float3 res;
    if (type == 0)
        res = make_float3(CalculateRadiance(ori[idx], dir[idx], lightDir, lightColor, alpha, multiScatter, g, sampleNum));
    else if (type == 1)
        res = make_float3(NNPredict<Type::RPNN>(ori[idx], dir[idx], lightDir, lightColor, alpha, g));
    else
        res = make_float3(NNPredict<Type::MRPNN>(ori[idx], dir[idx], lightDir, lightColor, alpha, g));

    result[idx] = res;

    if (threadIdx.x == 0)
        atomicAdd((int*)record, 1);
}

__global__ void GetSampleMulti(volatile int* record, int task_num, float3* result, float* alpha, float3* ori, float3* dir, float3* lightDir, float* g, float* scatters, float3 lightColor = { 1, 1, 1 }, int multiScatter = 1, int sampleNum = 1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= task_num) return;

    float3 res = GetSample(ori[idx], dir[idx], lightDir[idx], lightColor, scatters[idx], alpha[idx], multiScatter, g[idx], sampleNum);

    result[idx] = res;

    if (threadIdx.x == 0)
        atomicAdd((int*)record, 1);
}

__device__ int dev_checkboard = 0;
__device__ int flip = 0;

__device__ float exposure = 1;

__device__ float3 lori;
__device__ float3 lup;
__device__ float3 lright;
template<bool predict, int type = Type::MRPNN>
__global__ void RenderCamera(float3* target, Histogram* histo_buffer, int2 size, float3 ori, float3 up, float3 right, float3 lightDir, float3 lightColor, float alpha, int multiScatter, float g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    if (dev_checkboard) j = (blockIdx.y * blockDim.y + threadIdx.y) * 2 + ((i + flip) % 2);
    else j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = i * size.x + j;

    curandState seed;
    InitRand(&seed);

    float u = 1 - (j + Rand(&seed)) / size.x;
    float v = (i + Rand(&seed)) / size.y;

    float3 forward = normalize(-ori);
    float3 dir = forward + (right * (u * 2 - 1)) + (up * (v * 2 - 1));
    dir = normalize(dir);

    float4 res_dis;
    if (predict)
        res_dis = NNPredict<type>(ori, dir, lightDir, lightColor, alpha, g);
    else
        res_dis = CalculateRadiance(ori, dir, lightDir, lightColor, alpha, multiScatter, g, 1);
    float3 res = make_float3(res_dis);
    bool sky = res_dis.w < 0;
    float dis = max(0.001f, res_dis.w < 0 ? 10.0f : res_dis.w);

    res = max(float3{ 0 }, res);


    // show Lut
    {
        //float aaa = tex2D<float>(_HGLut, 1.2 * u - 0.1, 1.2 * v - 0.1);
        //res = { aaa,aaa,aaa };

        //if (abs(abs(1.2 * u - 0.1 - 0.5) - 0.5) < 0.001 || abs(abs(1.2 * v - 0.1 - 0.5) - 0.5) < 0.001) {
        //    res = { 1, 0, 1 };
        //}
    }

    if (i >= size.x || j >= size.y) return;


    int fNum = dev_checkboard ? frameNum / 2 : frameNum;
    if (!predict) {

        if (fNum == 0)
            histo_buffer[idx] = { 0 };

        float lerp_rate = 1.0f / (1 + (fNum));
        target[idx] = lerp(target[idx], res, lerp_rate);

        res = res / (res + 1);

        histo_buffer[idx].totalSampleNum += 1;

        int3 bin_idx = floor(min(res, float3{ 0.999f, 0.999f, 0.999f }) * HISTO_SIZE);

        histo_buffer[idx].bin[bin_idx.x] += 1;
        histo_buffer[idx].bin[bin_idx.y + HISTO_SIZE] += 1;
        histo_buffer[idx].bin[bin_idx.z + HISTO_SIZE * 2] += 1;

        float l = dot(res, { 1, 1, 1 });

        histo_buffer[idx].x = lerp(histo_buffer[idx].x, l, 1.0f / (1 + fNum));
        histo_buffer[idx].x2 = lerp(histo_buffer[idx].x2, l * l, 1.0f / (1 + fNum));
    }
    else {

        int lidx;
        {   // reprojection
            float3 motion_pos = ori + dir * dis;
            float3 ldir = motion_pos - lori;
            float3 lforward = normalize(lori);
            ldir = ldir / dot(ldir, lforward);
            ldir = ldir - lforward;
            float lu = dot(ldir, lright) * 0.5 + 0.5;
            float lv = dot(ldir, lup) * -0.5 + 0.5;
            int2 lxy = int2{ min(size.x - 1, max(0, int(lu * size.x))), min(size.y - 1, max(0, int(lv * size.y))) };
            lidx = lxy.y * size.x + lxy.x;
        }

        float lerp_rate = 1.0f / (1 + fNum);
        if (!sky && !dev_checkboard) lerp_rate = min(0.2f, lerp_rate);
        float3 his = float3{ histo_buffer[lidx].totalSampleNum,histo_buffer[lidx].x, histo_buffer[lidx].x2 };
        target[idx] = lerp(his, res, lerp_rate);
    }
}

template<bool predict, int type = Type::MRPNN>
__global__ void RenderCamera(float3* result, int2 size, float3 ori, float3 up, float3 right, float3 lightDir, float3 lightColor, float alpha, int multiScatter, float g) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = (blockIdx.y * blockDim.y + threadIdx.y);

    int idx = i * size.x + j;

    curandState seed;
    InitRand(&seed);

    float u = 1 - (j + Rand(&seed)) / size.x;
    float v = 1 - (i + Rand(&seed)) / size.y;

    float3 forward = normalize(-ori);
    float3 dir = forward + (right * (u * 2 - 1)) + (up * (v * 2 - 1));
    dir = normalize(dir);

    float4 res_dis;
    if (predict)
        res_dis = NNPredict<type>(ori, dir, lightDir, lightColor, alpha, g);
    else
        res_dis = CalculateRadiance(ori, dir, lightDir, lightColor, alpha, multiScatter, g, 1);
    float3 res = make_float3(res_dis);
    float dis = max(0.001f, res_dis.w < 0 ? 10.0f : res_dis.w);

    res = max(float3{ 0 }, res);

    if (i >= size.x || j >= size.y) return;

    float lerp_rate = 1.0f / (1 + frameNum);
    result[idx] = lerp(result[idx], res, lerp_rate);
}

__device__ int UnRoll(int2 idx, int2 wh) {
    idx.y = min(max(0, idx.y), wh.x - 1);
    idx.x = min(max(0, idx.x), wh.y - 1);
    return idx.y + wh.x * idx.x;
}

__device__ float Compare(Histogram x, Histogram y) {
    float nx = x.totalSampleNum;
    float ny = y.totalSampleNum;
    float sqrt_y_x = sqrt(ny / nx);
    float sqrt_x_y = sqrt(nx / ny);

    float p = 0;
    float res = 0;
    for (int i = 0; i < HISTO_SIZE * 3; i++)
    {
        float hx = x.bin[i];
        float hy = y.bin[i];
        if (hx != 0 || hy != 0) {
            p++;
            float t = sqrt_y_x * hx - sqrt_x_y * hy;
            res += t * t / (hx + hy);
        }
    }
    return p == 0 ? 0 : res / p;
}

template<bool denoise>
__global__ void Denoise(float3* target, Histogram* histo_buffer, unsigned int* target2, int2 size, int toneType) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size.x * size.y) return;

    int2 idx = int2{ id / size.x, id % size.x };

    float3 res;

    if (frameNum == 0) {
        if ((((idx.x + flip) % 2) + idx.y) % 2 == 0) {
            res = target[id];
        }
        else {
            res = target[UnRoll({ idx.x - 1, idx.y }, size)] + target[UnRoll({ idx.x + 1, idx.y }, size)]
                    + target[UnRoll({ idx.x, idx.y - 1 }, size)] + target[UnRoll({ idx.x, idx.y + 1 }, size)];
            res = res / 4;
        }
    }
    else {
        float variance = abs(histo_buffer[id].x2 - histo_buffer[id].x * histo_buffer[id].x);

        if (denoise && variance > 0.01) {
            Histogram center = histo_buffer[UnRoll(idx, size)];
            res = { 0 };
            float ws = 0;
            for (int i = -5; i <= 5; i++)
            {
                for (int j = -5; j <= 5; j++)
                {
                    int2 pairId = { idx.x + i, idx.y + j };
                    int t = UnRoll(pairId, size);
                    float w = max(0.0f, 1.0f - 1.2 * Compare(center, histo_buffer[t]));
                    res = res + target[t] * w;
                    ws += w;
                }
            }
            res = res / ws;
        }
        else {
            res = target[id];
        }
    }

    float3 val = res * exposure;

    if (toneType == 1)
        val = Gamma(val);
    else if (toneType == 2)
        val = ACES(val);

    const unsigned int red = (unsigned int)(255.0f * saturate_(val.x));
    const unsigned int gre = (unsigned int)(255.0f * saturate_(val.y));
    const unsigned int blu = (unsigned int)(255.0f * saturate_(val.z));
    target2[id] = 0xff000000 | (red << 16) | (gre << 8) | blu;
}

__global__ void ReprojectionDenoise(float3* target, Histogram* histo_buffer, unsigned int* target2, int2 size, int toneType) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size.x * size.y) return;

    histo_buffer[id].totalSampleNum = target[id].x;
    histo_buffer[id].x = target[id].y;
    histo_buffer[id].x2 = target[id].z;
    float3 res = target[id];

    float3 val = res * exposure;

    if (toneType == 1)
        val = Gamma(val);
    else if (toneType == 2)
        val = ACES(val);

    const unsigned int red = (unsigned int)(255.0f * saturate_(val.x));
    const unsigned int gre = (unsigned int)(255.0f * saturate_(val.y));
    const unsigned int blu = (unsigned int)(255.0f * saturate_(val.z));
    target2[id] = 0xff000000 | (red << 16) | (gre << 8) | blu;
}

__global__ void ClearHis(Histogram* histo_buffer, int2 size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size.x * size.y) return;

    histo_buffer[id] = { 0 };
}

__global__ void CalculateShadowTerm_TR(float3* result, float3 ori, float3 dir, float3 lightDir, float alpha = 1, float g = 0) {

    float3 res = ShadowTerm_TR(ori, dir, lightDir, alpha, g);

    result[blockIdx.x * blockDim.x + threadIdx.x] = res;
}

float3 VolumeRender::GetTr(float3 ori, float3 dir, float3 lightDir, float alpha,float g, int sampleNum) const 
{
    int group = 32;
    int group_num = sampleNum / group + (sampleNum % group != 0 ? 1 : 0);
    float3* results;
    cudaMalloc(&results, sizeof(float3) * group_num * group);
    CheckError;
    CalculateShadowTerm_TR << <group_num, group >> > (results, ori, dir, lightDir,  alpha, g);
    float3* res_cpu = new float3[group_num * group];
    cudaDeviceSynchronize();
    CheckError;
    cudaMemcpy(res_cpu, results, sizeof(float3) * group * group_num, cudaMemcpyDeviceToHost);
    CheckError;
    cudaFree(results);
    CheckError;
    float3 res = { 0,0,0 };
    for (int i = 0; i < group * group_num; i++)
    {
        res = res + res_cpu[i];
    }
    delete[]res_cpu;
    return res / (group * group_num);
}

vector<float3> VolumeRender::GetRadiances(vector<float3> ori, vector<float3> dir, float3 lightDir, float3 lightColor, float alpha, int multiScatter, float g, int sampleNum, RenderType rt) {

    if (rt != RenderType::PT) {
        UpdateHGLut(g);
        Update_TR(lightDir, alpha);
    }

    int task_num = ori.size();

    int group = 32;
    int group_num = task_num / group + (task_num % group != 0 ? 1 : 0);

    float3* results;
    cudaMalloc(&results, sizeof(float3) * task_num);
    float3* oris;
    cudaMalloc(&oris, sizeof(float3) * task_num);
    float3* dirs;
    cudaMalloc(&dirs, sizeof(float3) * task_num);

    CheckError;

    cudaMemcpy(oris, ori.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);

    CheckError;

    cudaMemcpy(dirs, dir.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);
    
    CheckError;
    
    volatile int* d_rec, *h_rec;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void**)&h_rec, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_rec, (int*)h_rec, 0);
    *h_rec = 0;
    if (rt == RenderType::PT)
        CalculateRadianceMulti<0><<<group_num, group>>>(d_rec, results, oris, dirs, lightDir, lightColor, alpha, multiScatter, g, sampleNum);
    else if (rt == RenderType::RPNN)
        CalculateRadianceMulti<1><<<group_num, group>>>(d_rec, results, oris, dirs, lightDir, lightColor, alpha, multiScatter, g, sampleNum);
    else        
        CalculateRadianceMulti<2><<<group_num, group>>>(d_rec, results, oris, dirs, lightDir, lightColor, alpha, multiScatter, g, sampleNum);
    
    auto call_back = thread([&](){
        int value = 0;
        do {
            int value1 = *h_rec;
            if (value1 > value) {
                printf("Rendering: %6.2f%%\n", value1 * 100.0f / group_num);
                value = value1;
            }
            wait(1000);
        } while (value < group_num);
    });
    
    cudaDeviceSynchronize();

    call_back.join();

    cudaFreeHost((void*)h_rec);

    CheckError;

    vector<float3> res_cpu(task_num);

    cudaMemcpy(res_cpu.data(), results, sizeof(float3) * task_num, cudaMemcpyDeviceToHost);

    CheckError;

    cudaFree(results);
    cudaFree(oris);
    cudaFree(dirs);

    CheckError;

    return res_cpu;
}

int flip_cpu = 0;
int rand_cpu = 0;

float3 last_ori, last_up, last_right;


vector<float3> VolumeRender::Render(int2 size, float3 ori, float3 up, float3 right, float3 lightDir, RenderType rt, float g, float alpha, float3 lightColor, int multiScatter, int sampleNum) {

    float3* results;
    cudaMalloc(&results, size.x * size.y * sizeof(float3));

    for (int i = 0; i < sampleNum; i++)
    {
        if (env_tex_dev != NULL && (rt != RenderType::PT)) {

            float rate = hdri_exp * 4 / (hdri_exp * 4 + max(lightColor.x, max(lightColor.y, lightColor.z)));

            if (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) < rate) {
                float3 rnd = Roberts2(rand_cpu);
                float3 dir = UniformSampleSphere(float2{ rnd.x, rnd.y });

                float2 uv = float2{ atan2f(-dir.z, dir.x) * (float)(0.5 / 3.1415926) + 0.5f, acosf(fmaxf(fminf(dir.y, 1.0f), -1.0f)) * (float)(1.0 / 3.1415926) };

                lightDir = dir;
                lightColor = hdri_img.Sample(uv) * hdri_exp * 4 / rate;
            }
            else
                lightColor = lightColor / (1 - rate);
        }

        if (rt != RenderType::PT) {
            UpdateHGLut(g);
            Update_TR(lightDir, alpha);
        }

        cudaMemcpyToSymbol(frameNum, &i, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(randNum, &i, sizeof(int), 0, cudaMemcpyHostToDevice);

        dim3 dimBlock(8, 4);
        dim3 dimGrid;

        dimGrid.x = (size.x + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = (size.y + dimBlock.y - 1) / dimBlock.y;
         
        if (rt == RenderType::PT)
            RenderCamera<false><<<dimGrid, dimBlock>>>(results, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);
        else if (rt == RenderType::RPNN)
            RenderCamera<true, Type::RPNN><<<dimGrid, dimBlock>>>(results, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);
        else 
            RenderCamera<true, Type::MRPNN><<<dimGrid, dimBlock>>>(results, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);

        cudaDeviceSynchronize();

        CheckError;

    }

    vector<float3> res_cpu(size.x * size.y);

    cudaMemcpy(res_cpu.data(), results, sizeof(float3) * size.x * size.y, cudaMemcpyDeviceToHost);
    cudaFree(results);

    return res_cpu;
}

void VolumeRender::Render(float3* target, Histogram* histo_buffer, unsigned int* target2, int2 size, float3 ori, float3 up, float3 right, float3 lightDir, float3 lightColor, float alpha, int multiScatter, float g, int randseed, RenderType rt, int toneType, bool denoise) {

    if (env_tex_dev != NULL && (rt != RenderType::PT)) {

        float rate = hdri_exp * 4 / (hdri_exp * 4 + max(lightColor.x, max(lightColor.y, lightColor.z)));

        if (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) < rate) {
            float3 rnd = Roberts2(rand_cpu);
            float3 dir = UniformSampleSphere(float2{ rnd.x, rnd.y });

            float2 uv = float2{ atan2f(-dir.z, dir.x) * (float)(0.5 / 3.1415926) + 0.5f, acosf(fmaxf(fminf(dir.y, 1.0f), -1.0f)) * (float)(1.0 / 3.1415926) };

            lightDir = dir;
            lightColor = hdri_img.Sample(uv) * hdri_exp * 4 / rate;
        }
        else
            lightColor = lightColor / (1 - rate);
    }

    if (rt != RenderType::PT) {
        UpdateHGLut(g);
        Update_TR(lightDir, alpha);
    }

    cudaMemcpyToSymbol(frameNum, &randseed, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(randNum, &rand_cpu, sizeof(int), 0, cudaMemcpyHostToDevice);    
    cudaMemcpyToSymbol(flip, &flip_cpu, sizeof(int), 0, cudaMemcpyHostToDevice);

    flip_cpu = (flip_cpu + 1) % 2;    
    rand_cpu++;

    dim3 dimBlock(8, 4);
    dim3 dimGrid;

    dimGrid.x = (size.x + dimBlock.x - 1) / dimBlock.x;
    if (checkboard)
        dimGrid.y = (size.y + dimBlock.y * 2 - 1) / (dimBlock.y * 2);
    else
        dimGrid.y = (size.y + dimBlock.y) / dimBlock.y;

    int task_num = size.x * size.y;
    int group = 32;
    int group_num = task_num / group + (task_num % group != 0 ? 1 : 0);

    bool predict = rt != RenderType::PT;
    if (!last_predict && predict) {
        ClearHis<<<group_num, group>>>(histo_buffer, size);
    }
    last_predict = predict;

    if (rt == RenderType::PT)
        RenderCamera<false><<<dimGrid, dimBlock>>>(target, histo_buffer, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);
    else if (rt == RenderType::RPNN)
        RenderCamera<true, Type::RPNN><<<dimGrid, dimBlock>>>(target, histo_buffer, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);
    else
        RenderCamera<true, Type::MRPNN><<<dimGrid, dimBlock>>>(target, histo_buffer, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g);

    if (!predict) {
        if (denoise)
            Denoise<true><<<group_num, group>>>(target, histo_buffer, target2, size, toneType);
        else
            Denoise<false><<<group_num, group>>>(target, histo_buffer, target2, size, toneType);
    }
    else
        ReprojectionDenoise<<<group_num, group>>>(target, histo_buffer, target2, size, toneType);

    cudaMemcpyToSymbol(lori, &ori, sizeof(float3), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(lup, &up, sizeof(float3), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(lright, &right, sizeof(float3), 0, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    CheckError;

    last_ori = ori;
    last_up = up;
    last_right = right;
    return;
}


vector<float3> VolumeRender::GetSamples(vector<float> alpha, vector<float3> ori, vector<float3> dir, vector<float3> lightDir, vector<float> g, vector<float> scatter, float3 lightColor, int multiScatter, int sampleNum) const {

    int task_num = ori.size();

    int group = 32;
    int group_num = task_num / group + (task_num % group != 0 ? 1 : 0);

    float3* results;
    cudaMalloc(&results, sizeof(float3) * task_num);
    float3* oris;
    cudaMalloc(&oris, sizeof(float3) * task_num);
    float3* dirs;
    cudaMalloc(&dirs, sizeof(float3) * task_num);
    float3* ldirs;
    cudaMalloc(&ldirs, sizeof(float3) * task_num);
    float* as;
    cudaMalloc(&as, sizeof(float) * task_num);
    float* gs;
    cudaMalloc(&gs, sizeof(float) * task_num);
    float* scatters;
    cudaMalloc(&scatters, sizeof(float) * task_num);
    CheckError;

    cudaMemcpy(oris, ori.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dirs, dir.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(ldirs, lightDir.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(as, alpha.data(), sizeof(float) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(gs, g.data(), sizeof(float) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(scatters, scatter.data(), sizeof(float) * task_num, cudaMemcpyHostToDevice);

    CheckError;

    volatile int* d_rec, * h_rec;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void**)&h_rec, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_rec, (int*)h_rec, 0);
    *h_rec = 0;
    GetSampleMulti<<<group_num, group>>>(d_rec, task_num, results, as, oris, dirs, ldirs, gs, scatters, lightColor, multiScatter, sampleNum);

    auto call_back = thread([&]() {
        int value = 0;
        do {
            int value1 = *h_rec;
            if (value1 > value) {
                printf("Rendering: %6.2f%%\n", value1 * 100.0f / group_num);
                value = value1;
            }
            wait(1000);
        } while (value < group_num);
        });

    cudaDeviceSynchronize();

    call_back.join();

    cudaFreeHost((void*)h_rec);

    CheckError;

    vector<float3> res_cpu(task_num);

    cudaMemcpy(res_cpu.data(), results, sizeof(float3) * task_num, cudaMemcpyDeviceToHost);

    CheckError;

    cudaFree(results);
    cudaFree(oris);
    cudaFree(dirs);
    cudaFree(ldirs);
    cudaFree(as);
    cudaFree(gs);
    cudaFree(scatters);

    CheckError;

    return res_cpu;
}

__global__ void GetTrMulti(int task_num, float3* result, float alpha, float3* ori, float3* dir, float3 lightDir, float3 lightColor,float g = 0, int sampleNum = 1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= task_num) return;

    float3 res = ShadowTerm_TRs(ori[idx], dir[idx], lightDir, lightColor,alpha, g, sampleNum);
    result[idx] = res;
}
//这里的dir应该是 中心位置看向偏移位置
vector<float3> VolumeRender::GetTrs(float alpha, vector<float3> ori, vector<float3> dir, float3 lightDir,float3 lightColor, float g, int sampleNum) const {

    int task_num = ori.size();

    int group = 32;
    int group_num = task_num / group + (task_num % group != 0 ? 1 : 0);

    float3* results;
    cudaMalloc(&results, sizeof(float3) * task_num);
    float3* oris;
    cudaMalloc(&oris, sizeof(float3) * task_num);
    float3* dirs;
    cudaMalloc(&dirs, sizeof(float3) * task_num);

    CheckError;

    cudaMemcpy(oris, ori.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dirs, dir.data(), sizeof(float3) * task_num, cudaMemcpyHostToDevice);

    CheckError;

    GetTrMulti<<<group_num, group>>>(task_num, results, alpha, oris, dirs, lightDir, lightColor,g, sampleNum);

    cudaDeviceSynchronize();
    CheckError;

    vector<float3> res_cpu(task_num);

    cudaMemcpy(res_cpu.data(), results, sizeof(float3) * task_num, cudaMemcpyDeviceToHost);

    CheckError;

    cudaFree(results);
    cudaFree(oris);
    cudaFree(dirs);
    CheckError;

    return res_cpu;
}
inline float Sample(float* data, int res, int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= res || y >= res || z >= res) return 0;
    return max(0.0f, data[((x * res) + y) * res + z]);
}
inline float SampleClamp(float* data, int res, int x, int y, int z) {
    x = max(min(x,res-1),0);
    y = max(min(y,res-1),0);
    z = max(min(z,res-1),0);
    return max(0.0f, data[((x * res) + y) * res + z]);
}
inline float Sample(float* data, int res, float3 uv) {
    float3 pos = uv * res - 0.5;
    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);
    float3 w = pos - make_float3(x, y, z);

    return
        lerp(
            lerp(
                lerp(Sample(data, res, x, y, z), Sample(data, res, x, y, z + 1), w.z),
                lerp(Sample(data, res, x, y + 1, z), Sample(data, res, x, y + 1, z + 1), w.z),
                w.y),
            lerp(
                lerp(Sample(data, res, x + 1, y, z), Sample(data, res, x + 1, y, z + 1), w.z),
                lerp(Sample(data, res, x + 1, y + 1, z), Sample(data, res, x + 1, y + 1, z + 1), w.z),
            w.y),
        w.x);
}
inline float SampleClamp(float* data, int res, float3 uv) {
    float3 pos = uv * res - 0.5;
    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);
    float3 w = pos - make_float3(x, y, z);

    return
        lerp(
            lerp(
                lerp(SampleClamp(data, res, x, y, z), SampleClamp(data, res, x, y, z + 1), w.z),
                lerp(SampleClamp(data, res, x, y + 1, z), SampleClamp(data, res, x, y + 1, z + 1), w.z),
                w.y),
            lerp(
                lerp(SampleClamp(data, res, x + 1, y, z), SampleClamp(data, res, x + 1, y, z + 1), w.z),
                lerp(SampleClamp(data, res, x + 1, y + 1, z), SampleClamp(data, res, x + 1, y + 1, z + 1), w.z),
                w.y),
            w.x);
}

//#define TR_MUL 3.1415926535f
#define TR_MUL 1.0f
inline float Sample_TR(float* data, int res, float3 uv,float alpha,float3 lightDir) {
    const int MaxStep = 128;
    float3 ori = uv - 0.5;
    float dis = RayBoxDistance(ori, lightDir);
    float MaxStepInv = dis / MaxStep;
    float phase = 1.0;
    float3 Lpos = ori;
    float shadowdist = 0;
    for (int i = 0; i < MaxStep; i++)
    {
        Lpos = Lpos + lightDir * MaxStepInv;
        float lsample = Sample(data, res, Lpos+0.5f);
        shadowdist = shadowdist + lsample;
    }
    float shadowterm = exp(-shadowdist * alpha * MaxStepInv) * phase;
    return TR_MUL * shadowterm;//
}

struct InitWeight {
    InitWeight();
};

void VolumeRender::MallocMemory() {
    cudaFree(0);

    datas = new float[resolution * resolution * resolution];
    hglut = new float[LUT_SIZE * LUT_SIZE];
    channel_desc = cudaCreateChannelDesc<float>();
    size = cudaExtent{ (size_t)resolution, (size_t)resolution, (size_t)resolution };
    cudaMalloc3DArray(&datas_dev, &channel_desc, size);

    for (int i = 0; i < 9; i++) {
        int reso = 256 >> i;
        mips[i] = new float[reso * reso * reso];
        mip_size[i] = cudaExtent{ (size_t)reso, (size_t)reso, (size_t)reso };
        cudaMalloc3DArray(mips_dev + i, &channel_desc, mip_size[i]);
    }
    for (int i = 0; i < 8; i++) {
        int reso = 128 >> i;
        tr_mips[i] = new float[reso * reso * reso];
        tr_mip_size[i] = cudaExtent{ (size_t)reso, (size_t)reso, (size_t)reso };
        cudaMalloc3DArray(tr_mips_dev + i, &channel_desc, tr_mip_size[i], cudaArraySurfaceLoadStore);
    }

    cudaMallocArray(&hglut_dev, &channel_desc, LUT_SIZE, LUT_SIZE, cudaArraySurfaceLoadStore);
}

//InitWeight weight;

VolumeRender::VolumeRender(int resolution) : resolution(resolution) {
    MallocMemory();
}

VolumeRender::VolumeRender(string path) {

    if (FILE* file = fopen((path + ".bin").c_str(), "rb")) {
        fread(&resolution, sizeof(int), 1, file);
        MallocMemory();
        fread(datas, sizeof(float), resolution * resolution * resolution, file);
        fclose(file);
        Update();
        return;
    }

    string format = path.substr(path.size() - 3, 3);
    if (format == "txt") {
        FILE* f = fopen(path.c_str(), "r");
        fscanf(f, "%d", &resolution);
        int total = resolution * resolution * resolution;
        MallocMemory();
        int index = 0;
        int loop_num = (total / 8) + (total % 8 != 0 ? 1 : 0);
        while (index < loop_num) {
            fscanf(f, "%f %f %f %f %f %f %f %f", datas + index * 8,
                datas + index * 8 + 1,
                datas + index * 8 + 2,
                datas + index * 8 + 3,
                datas + index * 8 + 4,
                datas + index * 8 + 5,
                datas + index * 8 + 6,
                datas + index * 8 + 7);
            index++;
        }
        fclose(f);
    }
    else if (format == "vox") {
        resolution = 256;
        MallocMemory();
        ifstream infile(path);
        string line;
        int i = 0;
        float inv = 64.0 / 255.0;
        while (!infile.eof()) {
            getline(infile, line); 
            string firstc = line.substr(0, 1);
            if (firstc != std::string("w") && firstc != std::string("h") && firstc != std::string("d")) {
                if (i >= 256 * 256 * 256) break;
                if (i % (256 * 256 * 32) == 0) printf("Loading vox percent:%.2f%%\n", 100.0 * (float)i / (256 * 256 * 256));
                stringstream data(line);
                int d[64];
                data >> d[0] >> d[1] >> d[2] >> d[3] >> d[4] >> d[5] >> d[6] >> d[7]
                    >> d[8] >> d[9] >> d[10] >> d[11] >> d[12] >> d[13] >> d[14] >> d[15]
                    >> d[16] >> d[17] >> d[18] >> d[19] >> d[20] >> d[21] >> d[22] >> d[23]
                    >> d[24] >> d[25] >> d[26] >> d[27] >> d[28] >> d[29] >> d[30] >> d[31]
                    >> d[32] >> d[33] >> d[34] >> d[35] >> d[36] >> d[37] >> d[38] >> d[39]
                    >> d[40] >> d[41] >> d[42] >> d[43] >> d[44] >> d[45] >> d[46] >> d[47]
                    >> d[48] >> d[49] >> d[50] >> d[51] >> d[52] >> d[53] >> d[54] >> d[55]
                    >> d[56] >> d[57] >> d[58] >> d[59] >> d[60] >> d[61] >> d[62] >> d[63];
                for (int j = 0; j < 64; j++)
                    datas[i + j] = (float)d[j] * inv;
                i += 64;
            }
        }
        infile.close();
    }
    else
    {
        printf("File not found!\n");
        return;
    }

    Update();

    {
        FILE* file = fopen((path + ".bin").c_str(), "wb");
        fwrite(&resolution, sizeof(int), 1, file);
        fwrite(datas, sizeof(float), resolution * resolution * resolution, file);
        fclose(file);
    }
}

VolumeRender::~VolumeRender() {
    cudaFreeArray(datas_dev);
    cudaFreeArray(hglut_dev);
    delete[]datas;
    delete[]hglut;
    for (int i = 0; i < 9; i++) {
        delete[] mips[i];
        cudaFreeArray(mips_dev[i]);
    }
    for (int i = 0; i < 8; i++) {
        delete[] tr_mips[i];
        cudaFreeArray(tr_mips_dev[i]);
    }

    if (env_tex_dev != 0) {
        cudaFreeArray(env_tex_dev);
    }
    if (hdri_img.data != 0) {
        delete hdri_img.data;
    }

}

void VolumeRender::SetData(int x, int y, int z, float value) {
    datas[(x * resolution + y) * resolution + z] = value;
}

void VolumeRender::SetDatas(FillFunc func) {
    ParallelFill(datas, resolution, func);
}

__device__ static const float FloatOneMinusEpsilon = 0.99999994;
#define OneMinusEpsilon FloatOneMinusEpsilon
__device__ static const unsigned int PrimeTableSize = 46;
__device__ static const unsigned int Primes[PrimeTableSize] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 };
__device__ static const unsigned int PrimeSums[PrimeTableSize] = {
    0, 2, 5, 10, 17, 28, 41, 58, 77, 100, 129, 160, 197, 238, 281, 328, 381, 440, 501, 568, 639, 712, 791, 874,
    963, 1060, 1161, 1264, 1371, 1480, 1593, 1720, 1851, 1988, 2127, 2276, 2427, 2584, 2747, 2914, 3087, 3266, 3447, 3638, 3831, 4028,
};
__device__ static const unsigned int RadicalInversePermutations_[] = {
0 ,1 ,2 ,1 ,0 ,4 ,1 ,2 ,3 ,0 ,6 ,2 ,4 ,1 ,0 ,5 ,3 ,8 ,3 ,5 ,4 ,7 ,2 ,6 ,
9 ,1 ,10 ,0 ,4 ,6 ,9 ,0 ,5 ,3 ,10 ,8 ,11 ,2 ,1 ,12 ,7 ,12 ,3 ,0 ,13 ,14 ,5 ,7 ,
8 ,9 ,4 ,11 ,16 ,10 ,6 ,2 ,15 ,1 ,12 ,2 ,6 ,11 ,4 ,1 ,18 ,5 ,15 ,3 ,8 ,14 ,13 ,0 ,
7 ,16 ,17 ,9 ,10 ,5 ,13 ,6 ,1 ,0 ,18 ,3 ,17 ,14 ,4 ,10 ,21 ,2 ,20 ,15 ,11 ,19 ,8 ,7 ,
12 ,9 ,22 ,16 ,19 ,28 ,15 ,25 ,7 ,10 ,22 ,0 ,24 ,21 ,4 ,5 ,3 ,27 ,2 ,12 ,20 ,16 ,13 ,1 ,
6 ,26 ,14 ,23 ,11 ,8 ,18 ,17 ,9 ,6 ,13 ,28 ,8 ,12 ,26 ,23 ,21 ,19 ,22 ,0 ,9 ,20 ,4 ,18 ,
3 ,27 ,25 ,7 ,29 ,15 ,30 ,1 ,14 ,5 ,11 ,24 ,2 ,16 ,17 ,10 ,12 ,8 ,18 ,9 ,15 ,10 ,27 ,36 ,
16 ,2 ,20 ,4 ,14 ,21 ,33 ,22 ,3 ,31 ,29 ,6 ,13 ,34 ,7 ,25 ,17 ,32 ,0 ,28 ,1 ,19 ,11 ,24 ,
5 ,26 ,30 ,23 ,35 ,3 ,25 ,32 ,0 ,35 ,19 ,11 ,21 ,10 ,20 ,16 ,26 ,18 ,17 ,1 ,38 ,5 ,39 ,31 ,
12 ,27 ,36 ,8 ,30 ,13 ,40 ,2 ,23 ,37 ,34 ,7 ,33 ,29 ,22 ,28 ,24 ,15 ,6 ,9 ,14 ,4 ,2 ,17 ,
34 ,33 ,28 ,30 ,20 ,14 ,23 ,6 ,16 ,4 ,1 ,21 ,38 ,26 ,13 ,41 ,39 ,0 ,7 ,31 ,3 ,32 ,40 ,8 ,
25 ,19 ,10 ,15 ,18 ,9 ,22 ,37 ,35 ,12 ,5 ,11 ,27 ,24 ,36 ,29 ,42 ,40 ,9 ,45 ,33 ,2 ,34 ,7 ,
13 ,37 ,32 ,30 ,11 ,38 ,19 ,44 ,17 ,16 ,4 ,10 ,27 ,6 ,25 ,21 ,12 ,15 ,31 ,3 ,8 ,35 ,46 ,28 ,
29 ,36 ,23 ,41 ,39 ,24 ,5 ,26 ,20 ,14 ,0 ,22 ,43 ,42 ,1 ,18 ,22 ,35 ,18 ,46 ,27 ,1 ,32 ,47 ,
52 ,10 ,49 ,17 ,13 ,6 ,43 ,28 ,0 ,31 ,48 ,45 ,37 ,15 ,41 ,33 ,8 ,30 ,39 ,36 ,40 ,38 ,50 ,29 ,
14 ,9 ,20 ,16 ,11 ,19 ,44 ,4 ,3 ,24 ,34 ,21 ,23 ,5 ,2 ,26 ,42 ,25 ,7 ,12 ,51 ,39 ,47 ,28 ,
17 ,49 ,10 ,56 ,44 ,21 ,58 ,13 ,33 ,51 ,16 ,46 ,23 ,53 ,15 ,32 ,27 ,12 ,34 ,52 ,45 ,40 ,4 ,8 ,
57 ,35 ,48 ,14 ,43 ,18 ,9 ,31 ,6 ,25 ,3 ,2 ,42 ,1 ,11 ,38 ,54 ,7 ,26 ,24 ,41 ,19 ,30 ,20 ,
0 ,5 ,50 ,29 ,55 ,37 ,36 ,22 ,33 ,1 ,21 ,22 ,34 ,4 ,52 ,55 ,6 ,13 ,11 ,56 ,44 ,30 ,43 ,16 ,
51 ,37 ,60 ,32 ,48 ,15 ,14 ,9 ,41 ,0 ,39 ,53 ,35 ,26 ,3 ,25 ,38 ,36 ,47 ,27 ,5 ,18 ,17 ,58 ,
54 ,7 ,20 ,59 ,19 ,49 ,50 ,2 ,23 ,12 ,40 ,31 ,45 ,57 ,46 ,28 ,10 ,24 ,29 ,8 ,42 ,30 ,18 ,28 ,
4 ,51 ,65 ,37 ,55 ,56 ,27 ,22 ,6 ,33 ,7 ,34 ,31 ,3 ,39 ,66 ,46 ,8 ,52 ,53 ,0 ,10 ,62 ,15 ,
13 ,43 ,5 ,14 ,48 ,40 ,36 ,63 ,20 ,58 ,19 ,11 ,24 ,2 ,25 ,59 ,54 ,61 ,42 ,32 ,38 ,64 ,44 ,49 ,
35 ,26 ,17 ,16 ,50 ,60 ,12 ,41 ,47 ,45 ,9 ,21 ,57 ,23 ,29 ,1 ,56 ,27 ,28 ,22 ,43 ,14 ,65 ,47 ,
20 ,35 ,55 ,60 ,53 ,7 ,57 ,52 ,69 ,63 ,3 ,30 ,10 ,23 ,0 ,58 ,67 ,2 ,17 ,50 ,41 ,31 ,40 ,15 ,
45 ,48 ,5 ,64 ,44 ,29 ,46 ,24 ,25 ,26 ,33 ,68 ,70 ,12 ,36 ,11 ,38 ,54 ,62 ,39 ,1 ,19 ,4 ,49 ,
16 ,6 ,21 ,59 ,61 ,18 ,32 ,9 ,8 ,34 ,13 ,42 ,66 ,51 ,37 ,52 ,10 ,16 ,51 ,44 ,3 ,29 ,9 ,71 ,
37 ,8 ,38 ,34 ,22 ,18 ,36 ,49 ,30 ,2 ,62 ,47 ,21 ,72 ,65 ,23 ,31 ,20 ,13 ,67 ,63 ,55 ,27 ,66 ,
32 ,15 ,26 ,56 ,0 ,48 ,17 ,6 ,60 ,50 ,12 ,57 ,41 ,46 ,24 ,40 ,42 ,1 ,11 ,39 ,54 ,35 ,33 ,64 ,
28 ,53 ,69 ,45 ,19 ,7 ,61 ,25 ,14 ,43 ,58 ,5 ,68 ,70 ,4 ,59 ,71 ,8 ,6 ,75 ,77 ,48 ,35 ,25 ,
13 ,36 ,1 ,57 ,63 ,60 ,21 ,31 ,17 ,20 ,28 ,78 ,72 ,2 ,51 ,46 ,44 ,30 ,37 ,42 ,65 ,18 ,32 ,16 ,
11 ,58 ,23 ,40 ,73 ,43 ,3 ,19 ,53 ,15 ,66 ,50 ,12 ,59 ,68 ,55 ,10 ,38 ,76 ,7 ,4 ,14 ,56 ,24 ,
70 ,61 ,0 ,45 ,69 ,33 ,27 ,62 ,39 ,9 ,67 ,47 ,41 ,29 ,34 ,49 ,26 ,22 ,52 ,74 ,54 ,64 ,5 ,57 ,
23 ,5 ,52 ,82 ,62 ,28 ,41 ,15 ,61 ,24 ,59 ,9 ,56 ,2 ,69 ,39 ,7 ,65 ,80 ,70 ,78 ,50 ,10 ,35 ,
11 ,72 ,81 ,31 ,42 ,60 ,66 ,54 ,19 ,25 ,22 ,63 ,44 ,18 ,14 ,46 ,74 ,13 ,76 ,8 ,12 ,29 ,53 ,38 ,
1 ,40 ,16 ,55 ,67 ,37 ,36 ,21 ,26 ,73 ,4 ,30 ,20 ,77 ,17 ,64 ,33 ,27 ,47 ,71 ,51 ,48 ,3 ,58 ,
32 ,45 ,6 ,49 ,75 ,68 ,34 ,43 ,0 ,79 ,8 ,58 ,45 ,78 ,28 ,52 ,26 ,65 ,84 ,15 ,83 ,14 ,10 ,20 ,
40 ,63 ,80 ,86 ,30 ,53 ,69 ,34 ,77 ,36 ,13 ,33 ,73 ,66 ,49 ,6 ,46 ,21 ,2 ,60 ,38 ,87 ,1 ,70 ,
43 ,61 ,27 ,79 ,44 ,4 ,71 ,74 ,64 ,82 ,23 ,39 ,57 ,32 ,37 ,51 ,16 ,25 ,81 ,50 ,68 ,17 ,29 ,75 ,
31 ,35 ,11 ,0 ,55 ,22 ,42 ,7 ,18 ,59 ,5 ,62 ,76 ,54 ,24 ,88 ,12 ,9 ,19 ,47 ,67 ,3 ,41 ,48 ,
56 ,85 ,72 ,64 ,30 ,1 ,75 ,81 ,59 ,19 ,28 ,65 ,2 ,88 ,0 ,21 ,89 ,96 ,70 ,37 ,73 ,24 ,94 ,43 ,
50 ,80 ,74 ,10 ,68 ,35 ,29 ,13 ,95 ,46 ,54 ,82 ,33 ,39 ,48 ,79 ,34 ,40 ,69 ,36 ,76 ,3 ,9 ,90 ,
41 ,92 ,42 ,55 ,77 ,91 ,87 ,84 ,26 ,52 ,18 ,61 ,51 ,15 ,44 ,56 ,32 ,8 ,25 ,66 ,27 ,57 ,11 ,45 ,
93 ,83 ,60 ,5 ,62 ,16 ,14 ,4 ,12 ,6 ,86 ,23 ,71 ,47 ,63 ,49 ,38 ,67 ,72 ,31 ,22 ,20 ,53 ,7 ,
85 ,78 ,17 ,58 ,51 ,0 ,32 ,14 ,54 ,21 ,90 ,12 ,36 ,94 ,28 ,4 ,66 ,65 ,77 ,56 ,61 ,52 ,84 ,16 ,
95 ,57 ,68 ,99 ,22 ,71 ,83 ,50 ,7 ,43 ,29 ,24 ,64 ,3 ,42 ,37 ,49 ,76 ,81 ,18 ,30 ,70 ,63 ,13 ,
20 ,58 ,53 ,74 ,79 ,5 ,38 ,31 ,45 ,67 ,86 ,98 ,100 ,73 ,69 ,40 ,62 ,17 ,1 ,10 ,23 ,96 ,78 ,55 ,
27 ,93 ,44 ,46 ,60 ,88 ,59 ,48 ,6 ,39 ,89 ,2 ,15 ,92 ,9 ,75 ,80 ,97 ,25 ,82 ,91 ,72 ,87 ,11 ,
34 ,26 ,41 ,85 ,35 ,19 ,33 ,47 ,8 ,70 ,77 ,43 ,34 ,65 ,75 ,5 ,2 ,7 ,17 ,88 ,39 ,97 ,12 ,67 ,
78 ,18 ,69 ,3 ,63 ,38 ,66 ,29 ,79 ,16 ,95 ,58 ,37 ,72 ,20 ,8 ,94 ,98 ,21 ,73 ,35 ,84 ,14 ,48 ,
25 ,85 ,13 ,90 ,92 ,59 ,9 ,71 ,99 ,31 ,96 ,42 ,33 ,46 ,30 ,83 ,26 ,23 ,11 ,28 ,91 ,60 ,93 ,82 ,
1 ,51 ,74 ,56 ,45 ,100 ,22 ,47 ,27 ,62 ,76 ,41 ,54 ,102 ,0 ,86 ,36 ,57 ,61 ,49 ,53 ,10 ,55 ,52 ,
40 ,64 ,44 ,87 ,6 ,101 ,4 ,81 ,50 ,80 ,32 ,19 ,68 ,24 ,89 ,15 ,46 ,76 ,22 ,78 ,41 ,27 ,97 ,61 ,
91 ,77 ,48 ,79 ,10 ,31 ,71 ,66 ,92 ,52 ,69 ,59 ,19 ,38 ,11 ,73 ,62 ,89 ,100 ,75 ,68 ,32 ,18 ,85 ,
95 ,3 ,84 ,45 ,105 ,72 ,64 ,14 ,29 ,1 ,26 ,90 ,96 ,25 ,54 ,43 ,0 ,20 ,17 ,7 ,55 ,80 ,39 ,21 ,
101 ,74 ,49 ,86 ,28 ,87 ,83 ,36 ,98 ,16 ,82 ,12 ,60 ,40 ,104 ,37 ,24 ,67 ,2 ,50 ,47 ,8 ,99 ,30 ,
35 ,15 ,94 ,56 ,51 ,88 ,53 ,5 ,106 ,6 ,57 ,44 ,58 ,70 ,9 ,4 ,13 ,103 ,42 ,81 ,65 ,102 ,34 ,33 ,
63 ,23 ,93 ,105 ,18 ,66 ,56 ,46 ,97 ,16 ,44 ,77 ,98 ,24 ,83 ,87 ,100 ,101 ,28 ,80 ,54 ,96 ,12 ,108 ,
43 ,81 ,78 ,29 ,45 ,69 ,11 ,104 ,95 ,52 ,10 ,21 ,8 ,67 ,73 ,63 ,25 ,93 ,32 ,70 ,71 ,38 ,88 ,89 ,
76 ,47 ,92 ,50 ,59 ,34 ,2 ,4 ,19 ,94 ,58 ,53 ,7 ,13 ,107 ,17 ,79 ,61 ,49 ,26 ,86 ,84 ,9 ,64 ,
82 ,85 ,91 ,31 ,20 ,99 ,51 ,1 ,14 ,42 ,65 ,30 ,23 ,74 ,106 ,68 ,15 ,0 ,41 ,33 ,22 ,75 ,103 ,6 ,
35 ,5 ,40 ,36 ,72 ,55 ,57 ,60 ,48 ,62 ,102 ,90 ,3 ,39 ,27 ,37 ,34 ,70 ,3 ,25 ,78 ,74 ,88 ,69 ,
103 ,48 ,33 ,92 ,57 ,61 ,80 ,111 ,39 ,29 ,81 ,45 ,96 ,32 ,38 ,101 ,55 ,66 ,46 ,43 ,99 ,49 ,8 ,22 ,
54 ,27 ,86 ,83 ,71 ,65 ,11 ,60 ,106 ,18 ,2 ,108 ,5 ,30 ,0 ,17 ,51 ,89 ,109 ,77 ,94 ,67 ,7 ,73 ,
95 ,35 ,37 ,87 ,20 ,90 ,13 ,50 ,26 ,23 ,9 ,85 ,1 ,36 ,104 ,63 ,68 ,15 ,12 ,100 ,47 ,64 ,52 ,107 ,
84 ,42 ,56 ,105 ,19 ,31 ,98 ,44 ,59 ,91 ,53 ,58 ,82 ,102 ,76 ,97 ,10 ,14 ,93 ,41 ,72 ,112 ,6 ,62 ,
16 ,79 ,40 ,24 ,4 ,75 ,28 ,110 ,21 ,6 ,17 ,115 ,112 ,100 ,11 ,126 ,65 ,119 ,97 ,0 ,92 ,81 ,32 ,35 ,
36 ,96 ,3 ,121 ,4 ,21 ,101 ,52 ,67 ,117 ,76 ,95 ,53 ,75 ,19 ,74 ,24 ,44 ,123 ,124 ,22 ,70 ,61 ,10 ,
38 ,71 ,29 ,46 ,69 ,66 ,28 ,80 ,125 ,48 ,60 ,99 ,2 ,82 ,64 ,55 ,59 ,13 ,87 ,72 ,85 ,57 ,73 ,26 ,
84 ,5 ,30 ,43 ,120 ,94 ,88 ,7 ,58 ,110 ,109 ,51 ,108 ,47 ,107 ,42 ,89 ,91 ,113 ,45 ,79 ,116 ,15 ,122 ,
63 ,98 ,16 ,20 ,118 ,77 ,31 ,25 ,50 ,68 ,33 ,78 ,56 ,1 ,23 ,41 ,9 ,37 ,12 ,34 ,27 ,102 ,83 ,111 ,
93 ,104 ,86 ,62 ,40 ,39 ,8 ,90 ,105 ,54 ,114 ,106 ,103 ,49 ,14 ,18 ,22 ,25 ,56 ,108 ,33 ,109 ,64 ,46 ,
116 ,92 ,2 ,39 ,32 ,96 ,120 ,110 ,35 ,80 ,97 ,77 ,62 ,14 ,117 ,85 ,70 ,73 ,6 ,81 ,59 ,57 ,50 ,41 ,
17 ,19 ,8 ,106 ,111 ,83 ,115 ,60 ,127 ,75 ,112 ,16 ,107 ,12 ,86 ,54 ,23 ,53 ,100 ,55 ,52 ,44 ,40 ,99 ,
5 ,78 ,10 ,30 ,9 ,114 ,49 ,79 ,69 ,113 ,18 ,128 ,34 ,20 ,0 ,122 ,47 ,4 ,129 ,71 ,1 ,88 ,13 ,74 ,
66 ,3 ,76 ,105 ,91 ,90 ,11 ,101 ,68 ,58 ,125 ,89 ,45 ,124 ,130 ,28 ,72 ,36 ,26 ,123 ,29 ,84 ,94 ,38 ,
103 ,24 ,67 ,65 ,21 ,61 ,51 ,126 ,93 ,102 ,42 ,98 ,43 ,7 ,121 ,63 ,31 ,15 ,118 ,95 ,87 ,48 ,104 ,119 ,
37 ,27 ,82 ,23 ,91 ,3 ,16 ,100 ,134 ,81 ,92 ,37 ,80 ,89 ,76 ,21 ,75 ,44 ,109 ,125 ,38 ,1 ,64 ,90 ,
124 ,111 ,56 ,83 ,7 ,82 ,6 ,85 ,40 ,39 ,66 ,33 ,108 ,127 ,57 ,46 ,19 ,36 ,115 ,69 ,10 ,50 ,55 ,95 ,
106 ,119 ,27 ,17 ,13 ,110 ,98 ,52 ,48 ,30 ,11 ,126 ,63 ,131 ,28 ,0 ,78 ,31 ,45 ,5 ,41 ,120 ,114 ,26 ,
86 ,113 ,77 ,135 ,14 ,47 ,61 ,96 ,136 ,58 ,87 ,133 ,104 ,9 ,105 ,79 ,59 ,42 ,103 ,67 ,123 ,43 ,15 ,2 ,
118 ,128 ,122 ,72 ,32 ,8 ,94 ,22 ,99 ,51 ,24 ,84 ,71 ,112 ,54 ,25 ,101 ,102 ,74 ,65 ,12 ,88 ,121 ,62 ,
97 ,116 ,107 ,68 ,29 ,35 ,130 ,34 ,73 ,132 ,70 ,117 ,18 ,49 ,4 ,60 ,20 ,129 ,53 ,93 ,36 ,74 ,39 ,24 ,
2 ,100 ,92 ,124 ,45 ,8 ,14 ,60 ,84 ,109 ,108 ,80 ,114 ,40 ,5 ,26 ,94 ,15 ,77 ,32 ,71 ,79 ,50 ,66 ,
119 ,61 ,46 ,129 ,47 ,68 ,116 ,10 ,55 ,53 ,135 ,128 ,16 ,93 ,38 ,113 ,65 ,9 ,87 ,48 ,43 ,54 ,105 ,95 ,
19 ,118 ,89 ,102 ,56 ,22 ,70 ,28 ,
};
__device__ unsigned int RadicalInversePermutations(unsigned int index) {
    return RadicalInversePermutations_[index];
}
__device__ float RadicalInverseSpecialized(unsigned int base, unsigned int a) {
    const float invBase = (float)1 / (float)base;
    unsigned int reversedDigits = 0;
    float invBaseN = 1;
    while (a) {
        float next = a / base;
        float digit = a - next * base;
        reversedDigits = reversedDigits * base + digit;
        invBaseN *= invBase;
        a = next;
    }
    return min(reversedDigits * invBaseN, OneMinusEpsilon);
}
__device__ float ScrambledRadicalInverse(unsigned int baseIndex, unsigned int a) {
    const unsigned int base = Primes[baseIndex];
    const unsigned int offset = PrimeSums[baseIndex];
    const float invBase = (float)1 / (float)base;
    unsigned int reversedDigits = 0;
    float invBaseN = 1;
    while (a) {
        unsigned int next = a / base;
        unsigned int digit = a - next * base;
        reversedDigits = reversedDigits * base + RadicalInversePermutations(offset + digit);
        invBaseN *= invBase;
        a = next;
    }
    return min(
        invBaseN * (reversedDigits + invBase * RadicalInversePermutations(offset) / (1 - invBase)),
        OneMinusEpsilon);
}
__device__ float HaltonSampleDimension(unsigned int dim, unsigned int index) {
    return ScrambledRadicalInverse(dim, index);
}

surface<void, cudaSurfaceType2D> hgSurfRef;
__global__ void Fill_Hg(float g) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= LUT_SIZE || j >= LUT_SIZE) return;

    float costheta = (((float)i + 0.5) / LUT_SIZE) * 2.0 - 1.0;

    float aa = (((float)j + 0.5) / LUT_SIZE);
    float angle = aa * 3.1415926535 * 60.0 / 180.0;//从0到60度


    curandState seed;
    InitRand(&seed);
    float bn = Rand(&seed);

    float hg = 0.0;
    float Radius = tan(angle);
    float3 ViewDir = float3{ 1.0f,0.0f,0.0f };
    float3 Dir = float3{ costheta,sqrt(1.0f - costheta * costheta),0.0f };

    float mis_coef = 0;
    {
        float p = Radius * Radius - 1 + Dir.x * Dir.x;
        if (p > 0)
        {
            float sdotl = g > 0 ? Dir.x : -Dir.x;
            float w = sqrt(p);
            float d0 = max(0.0f, sdotl - w) / Radius;
            float d1 = (sdotl + w) / Radius;
            float d = d1 * d1 * d1 - d0 * d0 * d0;
            mis_coef = (d1 > d0) * d;
        }
        mis_coef = min(max(mis_coef, 1e-8f), 1.0f);
        float a = abs(g);
        a = a < 0.6 ? 0 : (a < 0.7 ? (a - 0.6) * 9 : (pow((a - 0.7) / 0.3, 0.05) * 0.1 + 0.9));
        mis_coef = 1 - 1 / (1 + a * mis_coef / (1 - a));
    }

    for (int k = 0; k < 1024; k++)
    {
        if (frac(HaltonSampleDimension(4, k) + bn) < mis_coef) {
            //float2 rnd = Roberts2(k);
            float3 rnd = frac(float3{ HaltonSampleDimension(0, k), HaltonSampleDimension(1, k), 0 } + bn);
            float3 l = SampleHenyeyGreenstein(rnd.x, rnd.y, ViewDir, g);

            float sdotl = dot(Dir, l);
            float p = Radius * Radius - 1 + sdotl * sdotl;
            if (p <= 0) continue;
            float w = sqrt(p);
            float d0 = sdotl - w;
            float d1 = (sdotl + w) / Radius;
            d0 = max(0.0f, d0) / Radius;
            float d = d1 * d1 * d1 - d0 * d0 * d0;
            hg += max(0.0f, d / 4);
        }
        else {
            float3 rnd = frac(float3{ HaltonSampleDimension(0, k), HaltonSampleDimension(1, k), HaltonSampleDimension(2, k) } + bn);
            float theta = rnd.x * 2.0 * 3.1415926535;
            float phi = acos(2.0 * rnd.y - 1.0);
            float r = pow(rnd.z, 1 / 3.0f);
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);
            float x = Radius * r * sinPhi * cosTheta;
            float y = Radius * r * sinPhi * sinTheta;
            float z = Radius * r * cosPhi;
            float3 RandomPoint = float3{ x, y, z } + Dir;
            float cos = dot(normalize(RandomPoint), ViewDir);
            hg += HenyeyGreenstein(cos, g);
        }
    }

    hg /= 1024;

    surf2Dwrite(hg, hgSurfRef, i * sizeof(float), j);
}
void VolumeRender::UpdateHGLut(float g)
{
    if (g == hginlut) return;

    cudaBindSurfaceToArray(hgSurfRef, hglut_dev);

    CheckError;

    Fill_Hg<<<dim3(LUT_SIZE / 8, LUT_SIZE / 8), dim3(8,8)>>>(g);

    CheckError;

    _HGLut.normalized = true;
    _HGLut.filterMode = cudaFilterModeLinear;
    _HGLut.addressMode[0] = cudaAddressModeClamp;
    _HGLut.addressMode[1] = cudaAddressModeClamp;
    _HGLut.addressMode[2] = cudaAddressModeClamp;
    cudaBindTextureToArray(_HGLut, hglut_dev);

    CheckError;

    hginlut = g;
}
float VolumeRender::GetHGLut(float cos, float angle)
{
    float u = cos * 0.5 + 0.5;
    float v = angle / (3.1415926535 * 60.0 / 180.0);
    //float uu = max(min(u * LUT_SIZE, LUT_SIZE), 0);
    //float vv = max(min(v * LUT_SIZE, LUT_SIZE), 0);
    int u0 = max(min(int(u * LUT_SIZE), LUT_SIZE - 1), 0);
    int v0 = max(min(int(v * LUT_SIZE), LUT_SIZE - 1), 0);
    int u1 = max(min(int(u * LUT_SIZE) + 1, LUT_SIZE - 1), 0);
    int v1 = max(min(int(v * LUT_SIZE) + 1, LUT_SIZE - 1), 0);
    float a00 = hglut[u0 + v0 * LUT_SIZE];
    float a10 = hglut[u1 + v0 * LUT_SIZE];
    float a01 = hglut[u0 + v1 * LUT_SIZE];
    float a11 = hglut[u1 + v1 * LUT_SIZE];
    float ax0 = lerp(a00, a10, frac(v * LUT_SIZE));
    float ax1 = lerp(a01, a11, frac(v * LUT_SIZE));
    float axx = lerp(ax0, ax1, frac(u * LUT_SIZE));
    return axx;
}
void VolumeRender::Update() {
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr((void*)datas, resolution * sizeof(float), resolution, resolution);
    copyParams.dstArray = datas_dev;
    copyParams.extent = make_cudaExtent(resolution, resolution, resolution);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    CheckError;
    _DensityVolume.normalized = true;
    _DensityVolume.filterMode = cudaFilterModeLinear;
    _DensityVolume.addressMode[0] = cudaAddressModeBorder;
    _DensityVolume.addressMode[1] = cudaAddressModeBorder;
    _DensityVolume.addressMode[2] = cudaAddressModeBorder;

    cudaBindTextureToArray(_DensityVolume, datas_dev, channel_desc);
    CheckError;
    float md = 0.00001;
    for (int i = 0; i < resolution * resolution * resolution; i++)
    {
        md = max(md, datas[i]);
    }
    max_density = md;
    cudaMemcpyToSymbol(Resolution, &resolution, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(maxDensity, &md, sizeof(float), 0, cudaMemcpyHostToDevice);
    CheckError;
    int res = 256;
    while (res * 2 < resolution) res <<= 1;

    float* source = datas;
    int source_res = resolution;
    for (;res > 256; res >>= 1)
    {
        float* temp = new float[res * res * res];

        ParallelFill(temp, res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
        });

        if (source != datas)
            delete[] source;
        source = temp;
        source_res = res;
    }
    for (int mip = 0; mip < 9; mip++)
    {
        res = 256 >> mip;

        ParallelFill(mips[mip], res, [&](int x, int y, int z, float u, float v, float w) {
            return Sample(source, source_res, float3{ u,v,w });
            });

        if (mip == 0 && source != datas)
            delete[] source;
        source = mips[mip];
        source_res = res;
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
        copyParams.dstArray = mips_dev[mip];
        copyParams.extent = make_cudaExtent(source_res, source_res, source_res);
        copyParams.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);

        CheckError;
    }
    #define BindMip(i)  Mip(i).normalized = true;\
                        Mip(i).filterMode = cudaFilterModeLinear;\
                        Mip(i).addressMode[0] = cudaAddressModeBorder;\
                        Mip(i).addressMode[1] = cudaAddressModeBorder;\
                        Mip(i).addressMode[2] = cudaAddressModeBorder;\
                        cudaBindTextureToArray(Mip(i), mips_dev[i], channel_desc);

    BindMip(0); BindMip(1); BindMip(2);
    BindMip(3); BindMip(4); BindMip(5);
    BindMip(6); BindMip(7); BindMip(8);

    #undef BindMip

    CheckError;
}

texture<float, cudaTextureType3D, cudaReadModeElementType>  texRef;
surface<void, cudaSurfaceType3D> surfRef;
__global__ void Fill_TR(int res, float alpha, float3 lightDir) {

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= res || y >= res || z >= res) return;

    float3 uv = (float3{ (float)x, (float)y, (float)z } + 0.5) / res;

    const int MaxStep = 128;
    float3 ori = uv - 0.5;
    float dis = RayBoxDistance(ori, lightDir);
    float MaxStepInv = dis / MaxStep;
    float phase = 1.0;
    float3 Lpos = ori;
    float shadowdist = 0;
    for (int i = 0; i < MaxStep; i++)
    {
        Lpos = Lpos + lightDir * MaxStepInv;
        float lsample = tex3D<float>(texRef, Lpos.z + 0.5, Lpos.y + 0.5, Lpos.x + 0.5);
        shadowdist = shadowdist + lsample;
    }
    float shadowterm = exp(-shadowdist * alpha * MaxStepInv) * phase;
    
    surf3Dwrite(TR_MUL * shadowterm, surfRef, z * sizeof(float), y, x);
}
void VolumeRender::Update_TR(float3 lightDir,float alpha, bool CPU) 
{

    if (lightDir.x == tr_lightDir.x && lightDir.y == tr_lightDir.y && lightDir.z == tr_lightDir.z && alpha == tr_alpha)
        return;
    tr_lightDir = lightDir;
    tr_alpha = alpha;

    if (CPU) {
        int res = 256;
        for (int tr_mip = 0; tr_mip < 8; tr_mip++)
        {
            float* source = nullptr;
            res = 128 >> tr_mip;
            int source_res = res;
            if (true)//(tr_mip == 0)//(true)//
            {
                source = mips[tr_mip + 1];
                //float alphaScale = float(2 << tr_mip);
                float alphaScale = pow(1.73f, tr_mip + 1.0f);

                //todo : float alphaScale = tr_mip == 0? 1.0:pow(1.41, tr_mip);

                ParallelFill(tr_mips[tr_mip], res, [&](int x, int y, int z, float u, float v, float w) {
                    return Sample_TR(source, source_res, float3{ u,v,w }, alpha / alphaScale, lightDir);
                    });
            }
            else
            {
                source = tr_mips[tr_mip - 1];
                ParallelFill(tr_mips[tr_mip], res, [&](int x, int y, int z, float u, float v, float w) {
                    return Sample(source, source_res << 1, float3{ u,v,w });
                    });
            }
            source = tr_mips[tr_mip];
            cudaMemcpy3DParms copyParams = { 0 };
            copyParams.srcPtr = make_cudaPitchedPtr((void*)source, source_res * sizeof(float), source_res, source_res);
            copyParams.dstArray = tr_mips_dev[tr_mip];
            copyParams.extent = make_cudaExtent(source_res, source_res, source_res);
            copyParams.kind = cudaMemcpyHostToDevice;
            cudaMemcpy3D(&copyParams);

            CheckError;
        }
    }
    else {
        int res = 256;
        for (int tr_mip = 0; tr_mip < 8; tr_mip++)
        {
            res = 128 >> tr_mip;
            float alphaScale = pow(1.73f, tr_mip + 1.0f);
             
            texRef.normalized = true;
            texRef.filterMode = cudaFilterModeLinear;
            texRef.addressMode[0] = cudaAddressModeBorder;
            texRef.addressMode[1] = cudaAddressModeBorder;
            texRef.addressMode[2] = cudaAddressModeBorder;
            cudaBindTextureToArray(texRef, mips_dev[tr_mip + 1]);
            cudaBindSurfaceToArray(surfRef, tr_mips_dev[tr_mip]);

            dim3 gourp_size = dim3(4, 4, 4);
            dim3 gourp_num = dim3((res + 3) / 4, (res + 3) / 4, (res + 3) / 4);

            Fill_TR<<<gourp_num, gourp_size>>>(res, alpha / alphaScale, lightDir);
        }
    }
#define BindTR_Mip(i)   TR_Mip(i).normalized = true;\
                                TR_Mip(i).filterMode = cudaFilterModeLinear;\
                                TR_Mip(i).addressMode[0] = cudaAddressModeClamp;\
                                TR_Mip(i).addressMode[1] = cudaAddressModeClamp;\
                                TR_Mip(i).addressMode[2] = cudaAddressModeClamp;\
                                cudaBindTextureToArray(TR_Mip(i), tr_mips_dev[i]);
    BindTR_Mip(0); BindTR_Mip(1); BindTR_Mip(2);
    BindTR_Mip(3); BindTR_Mip(4); BindTR_Mip(5);
    BindTR_Mip(6); BindTR_Mip(7);
#undef BindTR_Mip
    CheckError;

}

void VolumeRender::SetHDRI(string path) {
    unsigned int rx, ry;
    float* pixels;
    load_hdr_float4(&pixels, &rx, &ry, path.c_str());

    if (env_tex_dev != 0)
        cudaFreeArray(env_tex_dev);
    if (hdri_img.data != 0)
        delete hdri_img.data;
    
    hdri_img.data = reinterpret_cast<float4*>(pixels);
    hdri_img.sx = rx;
    hdri_img.sy = ry;

    const cudaChannelFormatDesc hdri_desc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&env_tex_dev, &hdri_desc, rx, ry);

    CheckError;

    cudaMemcpyToArray(env_tex_dev, 0, 0, pixels, rx * ry * sizeof(float4), cudaMemcpyHostToDevice);

    CheckError;

    _HDRI.normalized = true;
    _HDRI.filterMode = cudaFilterModeLinear;
    _HDRI.addressMode[0] = cudaAddressModeWrap;
    _HDRI.addressMode[1] = cudaAddressModeWrap;
    _HDRI.addressMode[2] = cudaAddressModeWrap;

    cudaBindTextureToArray(_HDRI, env_tex_dev, hdri_desc);

    CheckError;
}

void VolumeRender::SetEnvExp(float exp)
{
    hdri_exp = exp;
    cudaMemcpyToSymbol(enviroment_exp, &exp, sizeof(float), 0, cudaMemcpyHostToDevice);
}
void VolumeRender::SetCheckboard(bool checkboard)
{
    this->checkboard = checkboard;
    int tmp = checkboard ? 1 : 0;
    cudaMemcpyToSymbol(dev_checkboard, &tmp, sizeof(int), 0, cudaMemcpyHostToDevice);
}
void VolumeRender::SetTrScale(float scale)
{
    cudaMemcpyToSymbol(tr_scale, &scale, sizeof(float), 0, cudaMemcpyHostToDevice);
}

void VolumeRender::SetScatterRate(float rate)
{
    SetScatterRate({ rate,rate,rate });
}

void VolumeRender::SetScatterRate(float3 rate)
{
    cudaMemcpyToSymbol(scatter_rate, &rate, sizeof(float3), 0, cudaMemcpyHostToDevice);
}

void VolumeRender::SetExposure(float exp)
{
    cudaMemcpyToSymbol(exposure, &exp, sizeof(float), 0, cudaMemcpyHostToDevice);
}
void VolumeRender::SetSurfaceIOR(float ior)
{
    cudaMemcpyToSymbol(IOR, &ior, sizeof(float), 0, cudaMemcpyHostToDevice);
}

float VolumeRender::DensityAtPosition(int mip, float3 pos) {
    return Sample(mips[mip], 256 >> mip, pos + 0.5);
}
float VolumeRender::TrAtPosition(int mip, float3 pos, float3 lightDir) {
    if (pos.x < -0.5 || pos.y < -0.5 || pos.z < -0.5 || pos.x > 0.5 || pos.y > 0.5 || pos.z > 0.5)
    {
        float offset = RayBoxOffset(pos, lightDir);
        if (offset >= 0)
        {
            return SampleClamp(tr_mips[mip], 128 >> mip, pos + 0.5 + lightDir * offset);
        }
        else
        {
            return TR_MUL;
        }
    }
    else
    {
        return SampleClamp(tr_mips[mip], 128 >> mip, pos + 0.5);
    }
}

float VolumeRender::DensityAtPosition(float mip, float3 pos) {
    int a = int(mip);
    float w = mip - a;
    return lerp(DensityAtPosition(a, pos), DensityAtPosition(a + 1, pos), w);
}

float VolumeRender::DensityAtUV(int mip, float3 uv) {
    return Sample(mips[mip], 256 >> mip, uv);
}

float VolumeRender::DensityAtUV(float mip, float3 uv) {
    int a = int(mip);
    float w = mip - a;
    return lerp(DensityAtUV(a, uv), DensityAtUV(a + 1, uv), w);
}

//__global__ void A() {
//    printf("%f,%f", __half22float2(LD7B[0]).x, __half22float2(LD7B[0]).y);
//}

InitWeight::InitWeight() {
    //SetWeights();
    //CheckError;
}