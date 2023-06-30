#include "volume.hpp"
#include "camera.hpp"
#include "sample_method.hpp"
#include "GUI.hpp"
#include <chrono>
#include <random>
#include <iomanip>


float hash1()
{
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}
class DisneyDescriptor
{
public:

    class Layer
    {
    public:
        const static size_t SIZE_X = 5;
        const static size_t SIZE_Y = 5;
        const static size_t SIZE_Z = 9;
        const static size_t LAYER_SIZE = SIZE_Z * SIZE_Y * SIZE_X;

        /**
         * sampled from SIZE_Z × SIZE_Y × SIZE_X grid in an axis-aligned box
         * with [−1, −1, −1] and [1, 1, 3] being two opposing corners.
         */
         //uint8_t density[LAYER_SIZE];
        float density[LAYER_SIZE];
    };

    const static size_t LAYERS_CNT = 10;

    /**
     * Each layer's support is 2x bigger than the previous.
     */
    Layer layers[LAYERS_CNT];
    float Gamma = 0.0f;
    float Radiance = 0.0f;
};


DisneyDescriptor GetDisneyDesc(VolumeRender& volume, float3 uv, float3 v, float3 s, float alpha, float descSizeAtLevel0) {

    DisneyDescriptor descriptor;
    v = normalize(v);
    const float3 eZ = normalize(s);
    const float3 eX = normalize(cross(eZ, v));
    const float3 eY = cross(eX, eZ);
    descriptor.Gamma = acos(dot(v, eZ));

    const float3 origin = uv;

    // 0.5f so that [−1, −1, −1] and [1, 1, 3] are in two opposing corners
    float scale = 0.5f * descSizeAtLevel0;
    // -1 because there are two sample points between 0 and 1.
    float mipmapLevel = 0;

    for (size_t layerId = 0; layerId < DisneyDescriptor::LAYERS_CNT; layerId++)
    {
        float currentmipmapLevel = max(min(mipmapLevel - 1.0f, 9.0f), 0.0f); //-2都去到2x2了采样没意义，-1就到1x1了
        uint32_t sampleId = 0;
        for (int z = -2; z <= 6; z++)
        {
            for (int y = -2; y <= 2; y++)
            {
                for (int x = -2; x <= 2; x++)
                {
                    float3 offset = (eX * x + eY * y + eZ * z) * scale;
                    const float3 pos = origin + offset;

                    float density = volume.DensityAtUV(int(currentmipmapLevel + 0.001f), pos);

                    descriptor.layers[layerId].density[sampleId] = density * alpha / 64.0f;
                    sampleId++;
                }
            }
        }
        scale *= 2;
        mipmapLevel++;
    }

    return descriptor;
}

class SamplePoint
{
public:
    float3 Position;
    float3 ViewDir;
    float3 LightDir;
    float Alpha;
    float g;

    SamplePoint(float3 p, float3 v, float3 l, float a, float g_)
    {
        Position = p;
        ViewDir = v;
        LightDir = l;
        Alpha = a;
        g = g_;
    }
};

float3 hash31sphere()
{
    float3 Rands = float3{ hash1(),hash1() ,hash1() };
    float theta = 2 * 3.14159265358979 * Rands.x;
    float phi = acos(2 * Rands.y - 1.0);//-pi pi
    float3 fp = float3{ cos(theta) * sin(phi),sin(theta) * sin(phi),cos(phi) };
    return normalize(fp);
}
float3 hash3box(float scale = 0.01)
{
    float3 Rands = hash31sphere() * cbrt(hash1());
    return Rands * scale;
}
float RayBoxOffset_(float3 p, float3 dir)
{
    dir = inv(dir);
    float3 bmax = { 0.4999f, 0.4999f, 0.4999f };
    float3 to_axil_dis = -p * dir;
    float3 axil_to_face_dis = bmax * dir;

    float3 dis0 = to_axil_dis + axil_to_face_dis;
    float3 dis1 = to_axil_dis - axil_to_face_dis;

    float3 tmin = min(dis0, dis1);
    float3 tmax = max(dis0, dis1);

    float tmi = max(tmin.x, max(tmin.y, tmin.z));
    float tma = min(tmax.x, min(tmax.y, tmax.z));

    return tma >= tmi ? max(tmi, 0.0f) : -1;
}
float RayBoxDistance_(float3 p, float3 dir)
{
    dir = inv(dir);
    float3 bmax = { 0.5f, 0.5f, 0.5f };
    float3 to_axil_dis = -p * dir;
    float3 axil_to_face_dis = bmax * dir;

    float3 dis0 = to_axil_dis + axil_to_face_dis;
    float3 dis1 = to_axil_dis - axil_to_face_dis;

    float3 tmax = max(dis0, dis1);

    float tma = min(tmax.x, min(tmax.y, tmax.z));

    return tma;
}
bool DeterminateNextVertex(VolumeRender& CurrentVolume, float alpha, float g, float3 pos, float3 dir, float dis, float3* nextPos, float3* nextDir)
{
    float SMax = CurrentVolume.max_density * alpha;
    float t = 0;
    int loop_num = 0;
    while (loop_num++ < 10000)
    {
        float rk = hash1();
        t -= log(1 - rk) / SMax;

        if (t > dis)
        {
            *nextPos = { 0, 0, 0 };
            *nextDir = { 0, 0, 0 };
            return false;
        }
        else
        {
            rk = hash1();
            float density = CurrentVolume.DensityAtPosition(0, pos + (dir * t));
            float S = density * alpha;
            if (S / SMax > rk)
            {
                break;
            }
            if (density < 0)
            {
                t -= density;
            }
        }
    }
    *nextDir = SampleHenyeyGreenstein(hash1(), hash1(), dir, g);
    *nextPos = (dir * t) + pos;
    return true;
}
void MeanFreePathSample(VolumeRender& CurrentVolume, vector<SamplePoint>& Samples, float3 ori, float3 dir, float3 lightDir, int maxcount, float alpha, float g)
{
    dir = normalize(dir);
    lightDir = normalize(lightDir);

    float dis = RayBoxOffset_(ori, dir);
    if (dis < 0)
    {
        return;
    }
    {
        float3 samplePosition = ori + dir * dis;
        float3 rayDirection = dir;
        for (int i = 0; i < 4; i++)
        {
            float3 nextPos, nextDir;
            float dis = RayBoxDistance_(samplePosition, rayDirection);
            bool in_volume = DeterminateNextVertex(CurrentVolume, alpha, g, samplePosition, rayDirection, dis, &nextPos, &nextDir);

            if (!in_volume || Samples.size() >= maxcount)
            {
                return;
            }
            if (i == 0 || dot(samplePosition - nextPos, samplePosition - nextPos) > 1.0 / 64.0 || hash1() > 0.9)
            {
                Samples.push_back(SamplePoint(hash1() > 0.5 ? nextPos + hash3box(1.0 / 128.0) : nextPos, hash1() > 0.25 ? dir : rayDirection, lightDir, alpha, g));
            }
            samplePosition = nextPos;
            rayDirection = nextDir;
        }
    }
    return;
}

void GetDesiredCountSample(VolumeRender& CurrentVolume, vector<SamplePoint>& Samples, int Count, float density_min, float density_max)
{
    Samples.clear();
    int last_print = 0;
    int print_per = Count / 8;
    while (Samples.size() < Count)
    {
        if (Samples.size() / print_per > last_print)
        {
            printf("Getting Samples: %.5f%%\n", float(Samples.size()) / Count * 100.0f);
            last_print = Samples.size() / print_per;
        }
        float3 ori = hash31sphere() * 3.0f;
        float3 dir = normalize(hash31sphere() + normalize(-ori));
        float3 ldir = hash31sphere();
        float Alpha = lerp(density_min, density_max, hash1());//lerp(density_min, density_max, 0.5f);//;/*没有随机*/
        float g = 0.857f;//没有随机
        MeanFreePathSample(CurrentVolume, Samples, ori, dir, ldir, Count, Alpha, g);
    }
}

void DebugSamples(string vpath, string outpath, int count = 512, float alpha = 1.0, float alpha_max = 5.0)
{
    VolumeRender v(vpath);
    vector<SamplePoint> Samples;
    GetDesiredCountSample(v, Samples, count, alpha, alpha_max);
    std::ofstream outfile(outpath);
    for (SamplePoint& s : Samples)
    {
        outfile << setiosflags(ios::fixed) << s.Position.x << ",";
        outfile << setiosflags(ios::fixed) << s.Position.y << ",";
        outfile << setiosflags(ios::fixed) << s.Position.z << ",";
        outfile << setiosflags(ios::fixed) << s.ViewDir.x << ",";
        outfile << setiosflags(ios::fixed) << s.ViewDir.y << ",";
        outfile << setiosflags(ios::fixed) << s.ViewDir.z << ",";
        outfile << setiosflags(ios::fixed) << s.LightDir.x << ",";
        outfile << setiosflags(ios::fixed) << s.LightDir.y << ",";
        outfile << setiosflags(ios::fixed) << s.LightDir.z << ",";
        outfile << std::endl;
    }
}

int main()
{
    std::string DataPath = "D:/Pytorch/VolumetricNN-main/Release/Data/";
    std::string DataName = "DS_10000.csv";
    std::string RelativePath = "D:/Pytorch/volume-cudagui0408/Build/Release/Data/";
    vector<std::string> DataList;
    DataList.push_back("dense.512.txt");
    DataList.push_back("mediocris_high.512.txt");
    DataList.push_back("cumulus_humilis.512.txt");
    DataList.push_back("cumulus_congestus1.512.txt");
    vector<float> DensityMin;
    DensityMin.push_back(0.5f);
    DensityMin.push_back(0.5f);
    DensityMin.push_back(1.0f);
    DensityMin.push_back(0.5f);
    vector<float> DensityMax;
    DensityMax.push_back(6.0f);
    DensityMax.push_back(12.0f);
    DensityMax.push_back(40.0f);
    DensityMax.push_back(6.0f);

    const int CountAll = 10000;
    const int MiniLoopCount = 1;
    int Computed = 0;
    int CountPer = CountAll / DataList.size() / MiniLoopCount;

    vector<DisneyDescriptor> Data;
    std::ofstream outfile(DataPath + DataName);
    outfile << "# " << CountAll << "x" << "(5,5,9)" << "x" << "10 Layers" << std::endl;
    for (int l = 0; l < MiniLoopCount; l++)
    {
        for (int i = 0; i < DataList.size(); i++)
        {
            Data.clear();
            printf("Processing %.2f%%\n", 100.0 * float(Computed) / Computed);
            printf("Computing:%s\n", DataList[i].c_str());
            printf("Desired Size:%d\n", CountPer);
            std::string CurrentData = DataList[i];
            float CurrentDensityMin = DensityMin[i];
            float CurrentDensityMax = DensityMax[i];
            VolumeRender v(RelativePath + CurrentData);
            printf("Getting Mean Free Path Samples\n");
            vector<SamplePoint> Samples;
            GetDesiredCountSample(v, Samples, CountPer, CurrentDensityMin, CurrentDensityMax);
            vector<float3> SampleOris;
            vector<float3> SampleDirs;
            vector<float3> SampleLDirs;
            vector<float> SampleAlphas;
            vector<float> SampleGs;
            vector<float> SampleScatters;

            printf("RealDataSet Size:%d\n", Samples.size());
            for (int i = 0; i < Samples.size(); i++)
            {
                SamplePoint CurrentSample = Samples[i];
                SampleOris.push_back(CurrentSample.Position);
                SampleDirs.push_back(CurrentSample.ViewDir);
                SampleLDirs.push_back(CurrentSample.LightDir);
                SampleAlphas.push_back(CurrentSample.Alpha);
                SampleGs.push_back(CurrentSample.g);
                SampleScatters.push_back(1.0f);
            }
            float3 LightColor = { 1.0,1.0,1.0 };
            vector<float3> CurrentRadiances = v.GetSamples(SampleAlphas, SampleOris, SampleDirs, SampleLDirs, SampleGs, SampleScatters, LightColor, 512, 1024);
            printf("RealRadianceSet Size:%d\n", CurrentRadiances.size());
            float gap = 0.25f / 1024.0f;
            for (int i = 0; i < CurrentRadiances.size(); i++)
            {
                DisneyDescriptor desc = GetDisneyDesc(v, SampleOris[i] + float3{ 0.5f, 0.5f, 0.5f }, SampleDirs[i], SampleLDirs[i], SampleAlphas[i], gap);
                desc.Radiance = CurrentRadiances[i].x;
                Data.push_back(desc);
            }
            printf("FinalDescSet Size:%d\n", Data.size());

            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);
            std::shuffle(Data.begin(), Data.end(), e);

            const int samplecount = 5 * 5 * 9;
            for (int i = 0; i < Data.size(); i++)
            {
                if (i % (Data.size() / 8) == 0)
                {
                    printf("Output Shuffle_Dataset:%.2f%%\n", 100.0f * (float)i / Data.size());
                }
                DisneyDescriptor& CS = Data[i];
                for (int j = 0; j < 10; j++)
                {
                    for (int k = 0; k < samplecount; k++)
                    {
                        outfile << setiosflags(ios::fixed) << CS.layers[j].density[k] << ",";
                    }
                }
                outfile << setiosflags(ios::fixed) << CS.Gamma << ",";
                outfile << setiosflags(ios::fixed) << CS.Radiance << std::endl;
                Computed++;
            }
        }
    }
    outfile.close();

    return 0;
}