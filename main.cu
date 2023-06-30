#include "volume.hpp"
#include "camera.hpp"
#include "sample_method.hpp"
#include "GUI.hpp"
#include <chrono>
#include <random>
#include <iomanip>

int main()
{
#if 1
    string cloud_path = "./TestCase/CLOUD0";
    VolumeRender v(cloud_path);
    float3 lightColor = { 1.0, 1.0, 1.0 };
    float alpha = 1;
    float3 CamPos = float3{ 0.67085, -0.03808, -0.04856 };

    Camera cam(v, "test camera");
    cam.resolution = 512;
    cam.SetPosition(CamPos);
    float g = 0.857;
    float3 scatter = float3{ 1, 1, 1 };
    v.SetScatterRate(scatter);
    float3 lightDir = normalize(float3{ 0.34281, 0.70711, 0.61845 });
#else
    VolumeRender v(512);
    v.SetDatas([](int x, int y, int z, float u, float v, float w) {
        float dis = distance(make_float3(0.5f, 0.5f, 0.5f), make_float3(u, v, w));
        return dis < 0.25 ? 1.0f : 0;
    });
    v.Update(); // Call Update after changing volumetric data.
    float3 lightColor = { 1.0, 1.0, 1.0 };
    float alpha = 2.0f;
    float3 CamPos = float3{ 0.67085, -0.03808, -0.04856 };

    Camera cam(v, "test camera");
    cam.resolution = 512;
    cam.SetPosition(CamPos);
    float g = 0.857;
    float3 scatter = float3{ 1, 1, 1 };
    v.SetScatterRate(scatter);
    float3 lightDir = normalize(float3{ 0.34281, 0.70711, 0.61845 });
#endif

    RunGUI(cam, v, lightDir, lightColor, scatter, alpha, 512, g);

    return 0;
}