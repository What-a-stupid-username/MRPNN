#pragma once

#include "volume.hpp"
#include <functional>

typedef function<float3(float3 radiance)> ToneMapFunction;

int SaveBMP(unsigned char* image, int imageWidth, int imageHeight, const char* filename);

class Camera {
protected:
    float3 ori;
    float3 forward;
    float3 right;
    float3 up;

    VolumeRender* volume;
public:

    static float3 Gamma(float3 color);

    static float3 ACES(float3 color);

    static float3 None(float3 color);
public:
    int resolution;
    string name;

    Camera(VolumeRender& volume, string name = "test camera", float3 position = { 0.75, 0.75, 0.75 }, int resolution = 512);

    void SetPosition(float3 position);

    float3 GetPosition() const { return ori; };

    void SetVolume(VolumeRender& volume);

    void Render(float3 lightDir, float3 lightColor = { 1, 1, 1 }, float alpha = 1, int multiScatter = 1, float g = 0, int sampleNum = 1, ToneMapFunction tone = ACES, VolumeRender::RenderType rt = VolumeRender::RenderType::PT, float exp = 1.0);

    void RenderToFile(string path, float3 lightDir, float3 lightColor = { 1, 1, 1 }, float alpha = 1, int multiScatter = 1, float g = 0, int sampleNum = 1, ToneMapFunction tone = ACES, VolumeRender::RenderType rt = VolumeRender::RenderType::PT, float exp = 1.0);

    void Render(float3* target, Histogram* histo_buffer, unsigned int* target2, int2 size, int frame, float3 lightDir, float3 lightColor = { 1, 1, 1 }, float alpha = 1, int multiScatter = 1, float g = 0, int tone = 2, VolumeRender::RenderType rt = VolumeRender::RenderType::PT, bool denoise = false);
}; 