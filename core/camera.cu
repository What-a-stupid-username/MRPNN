#pragma once

#include "camera.hpp"

#include "platform.h"
#include <time.h>

using namespace std;

int SaveBMP(unsigned char* image, int imageWidth, int imageHeight, const char* filename)
{
    unsigned char header[54] = {
        0x42, 0x4d, 0   , 0, 0, 0, 0   , 0,
        0   , 0   , 0x36, 0, 0, 0, 0x28, 0,
        0   , 0   , 0   , 0, 0, 0, 0   , 0,
        0   , 0   , 0x01, 0, 0x20, 0, 0, 0,
        0   , 0   , 0   , 0, 0x10, 0, 0, 0,
        0   , 0   , 0   , 0, 0   , 0, 0, 0,
        0   , 0   , 0   , 0, 0   , 0
    };

    for (int i = 0; i < imageWidth; i++)
    {
        for (int j = 0; j < imageHeight; j++)
        {
            int b = j + i * imageWidth;
            b *= 4;
            int k = image[b];
            image[b] = image[b + 2];
            image[b + 2] = k;
        }
    }

    long file_size = (long)imageWidth * (long)imageHeight * 4 + 54;
    header[2] = (unsigned char)(file_size & 0x000000ff);
    header[3] = (file_size >> 8) & 0x000000ff;
    header[4] = (file_size >> 16) & 0x000000ff;
    header[5] = (file_size >> 24) & 0x000000ff;

    long width = imageWidth;
    header[18] = width & 0x000000ff;
    header[19] = (width >> 8) & 0x000000ff;
    header[20] = (width >> 16) & 0x000000ff;
    header[21] = (width >> 24) & 0x000000ff;

    long height = -imageHeight;
    header[22] = height & 0x000000ff;
    header[23] = (height >> 8) & 0x000000ff;
    header[24] = (height >> 16) & 0x000000ff;
    header[25] = (height >> 24) & 0x000000ff;

    char fname_bmp[128];
    sprintf(fname_bmp, "%s.bmp", filename);

    FILE* fp;
    if (!(fp = fopen(fname_bmp, "wb")))
        return -1;

    fwrite(header, sizeof(unsigned char), 54, fp);
    fwrite(image, sizeof(unsigned char), (size_t)(long)imageWidth * imageHeight * 4, fp);

    fclose(fp);
    return 0;
}

float3 Camera::None(float3 color) {
    return color;
}

float3 Camera::Gamma(float3 color) {
    return pow(color, 1.0f / 2.2f);
}

float3 Camera::ACES(float3 color) {
    float3x3 AP1_2_XYZ_MAT = float3x3{ {0.6624541811, 0.1340042065, 0.1561876870},
                                       {0.2722287168, 0.6740817658, 0.0536895174},
                                       {-0.0055746495, 0.0040607335, 1.0103391003} };

    auto unity_to_ACES = [](float3 x)
    {
        float3x3 sRGB_2_AP0 = {
            {0.4397010, 0.3829780, 0.1773350},
            {0.0897923, 0.8134230, 0.0967616},
            {0.0175440, 0.1115440, 0.8707040}
        };
        x = sRGB_2_AP0 * x;
        return x;
    };
    auto ACES_to_ACEScg = [](float3 x)
    {
        float3x3 AP0_2_AP1_MAT = {
            {1.4514393161, -0.2365107469, -0.2149285693},
            {-0.0765537734,  1.1762296998, -0.0996759264},
            {0.0083161484, -0.0060324498,  0.9977163014}
        };
        return AP0_2_AP1_MAT * x;
    };
    auto XYZ_2_xyY = [](float3 XYZ)
    {
        float divisor = max(dot(XYZ, { 1,1,1 }), 1e-4);
        return float3{ XYZ.x / divisor, XYZ.y / divisor, XYZ.y };
    };
    auto xyY_2_XYZ = [](float3 xyY)
    {
        float m = xyY.z / max(xyY.y, 1e-4f);
        float3 XYZ = float3{ xyY.x, xyY.z, (1.0f - xyY.x - xyY.y) };
        XYZ.x *= m;
        XYZ.z *= m;
        return XYZ;
    };
    auto darkSurround_to_dimSurround = [&](float3 linearCV)
    {
        float3 XYZ = AP1_2_XYZ_MAT * linearCV;

        float3 xyY = XYZ_2_xyY(XYZ);
        xyY.z = min(max(xyY.z, 0.0), 65504.0);
        xyY.z = pow(xyY.z, 0.9811);
        XYZ = xyY_2_XYZ(xyY);

        float3x3 XYZ_2_AP1_MAT = {
            {1.6410233797, -0.3248032942, -0.2364246952},
            {-0.6636628587,  1.6153315917,  0.0167563477},
            {0.0117218943, -0.0082844420,  0.9883948585}
        };
        return XYZ_2_AP1_MAT * XYZ;
    };

    float3 aces = unity_to_ACES(color);

    float3 AP1_RGB2Y = float3{ 0.272229, 0.674082, 0.0536895 };

    float3 acescg = ACES_to_ACEScg(aces);
    float tmp = dot(acescg, AP1_RGB2Y);
    acescg = lerp(float3{ tmp,tmp,tmp }, acescg, 0.96);
    const float a = 278.5085;
    const float b = 10.7772;
    const float c = 293.6045;
    const float d = 88.7122;
    const float e = 80.6889;
    float3 x = acescg;
    float3 rgbPost = (x * (x * a + b)) / (x * (x * c + d) + e);
    float3 linearCV = darkSurround_to_dimSurround(rgbPost);
    tmp = dot(linearCV, AP1_RGB2Y);
    linearCV = lerp(float3{ tmp,tmp,tmp }, linearCV, 0.93);
    float3 XYZ = AP1_2_XYZ_MAT * linearCV;
    float3x3 D60_2_D65_CAT = {
        {0.98722400, -0.00611327, 0.0159533},
        {-0.00759836,  1.00186000, 0.0053302},
        {0.00307257, -0.00509595, 1.0816800}
    };
    XYZ = D60_2_D65_CAT * XYZ;
    float3x3 XYZ_2_REC709_MAT = {
        {3.2409699419, -1.5373831776, -0.4986107603},
        {-0.9692436363,  1.8759675015,  0.0415550574},
        {0.0556300797, -0.2039769589,  1.0569715142}
    };
    linearCV = XYZ_2_REC709_MAT * XYZ;

    return Gamma(linearCV);
}


Camera::Camera(VolumeRender& volume, string name, float3 position, int resolution) {
    ori = position;
    forward = normalize(-ori);
    right = normalize(cross({ 0,1,0 }, forward));
    up = cross(forward, right);
    this->name = name;
    this->volume = &volume;
    this->resolution = resolution;
}

void Camera::SetPosition(float3 position) {
    ori = position;
    forward = normalize(-ori);
    right = normalize(cross({ 0,1,0 }, forward));
    up = cross(forward, right);
    if (position.x == 0 && position.z == 0) {
        
        if (position.y > 0) {
            right = float3{1,0,0};
            up = float3{0,0,1};
        }
        else{
            right = float3{1,0,0};
            up = float3{0,0,-1};
        }
    }
}

void Camera::SetVolume(VolumeRender& volume) {
    this->volume = &volume;
}

void Camera::Render(float3 lightDir, float3 lightColor, float alpha, int multiScatter, float g, int sampleNum, ToneMapFunction tone, VolumeRender::RenderType rt, float exp) {

    unsigned char* data = new unsigned char[resolution * resolution * 4];

    int start_time = clock();

    auto ress = volume->Render(int2{ resolution, resolution }, ori, up, right, lightDir, rt, g, alpha, lightColor, multiScatter, sampleNum);

    printf("Rendering done in %.2f s.\n", (clock() - start_time) / 1000.0f);

    for (int i = 0; i < resolution; i++)
    {
        for (int j = 0; j < resolution; j++)
        {
            float3 res = tone(ress[i * resolution + j] * exp);
            res = saturate_(res);
            data[(i * resolution + j) * 4] = res.x * 255;
            data[(i * resolution + j) * 4 + 1] = res.y * 255;
            data[(i * resolution + j) * 4 + 2] = res.z * 255;
            data[(i * resolution + j) * 4 + 3] = 255;
        }
    }
    string affix = (rt == VolumeRender::RenderType::PT ? "_PT" : (rt == VolumeRender::RenderType::RPNN ? "_RPNN" : "MRPNN"));
    SaveBMP(data, resolution, resolution, ("./" + name + affix).c_str());

    OpenBMP(name + affix);

    delete[] data;
}


void Camera::RenderToFile(string path, float3 lightDir, float3 lightColor, float alpha, int multiScatter, float g, int sampleNum, ToneMapFunction tone, VolumeRender::RenderType rt, float exp) {

    unsigned char* data = new unsigned char[resolution * resolution * 4];

    int start_time = clock();

    auto ress = volume->Render(int2{ resolution, resolution }, ori, up, right, lightDir, rt, g, alpha, lightColor, multiScatter, sampleNum);

    printf("Rendering done in %.2f s.\n", (clock() - start_time) / 1000.0f);

    for (int i = 0; i < resolution; i++)
    {
        for (int j = 0; j < resolution; j++)
        {
            float3 res = tone(ress[i * resolution + j] * exp);
            res = saturate_(res);
            data[(i * resolution + j) * 4] = res.x * 255;
            data[(i * resolution + j) * 4 + 1] = res.y * 255;
            data[(i * resolution + j) * 4 + 2] = res.z * 255;
            data[(i * resolution + j) * 4 + 3] = 255;
        }
    }

    SaveBMP(data, resolution, resolution, path.c_str());

    delete[] data;
}


void Camera::Render(float3* target, Histogram* histo_buffer, unsigned int* target2, int2 size, int frame, float3 lightDir, float3 lightColor, float alpha, int multiScatter, float g, int tone, VolumeRender::RenderType rt, bool denoise) {
    volume->Render(target, histo_buffer, target2, size, ori, up, right, lightDir, lightColor, alpha, multiScatter, g, frame, rt, tone, denoise);
} 