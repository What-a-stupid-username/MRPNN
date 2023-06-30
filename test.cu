#include "volume.hpp"
#include "camera.hpp"
#include "sample_method.hpp"
#include "GUI.hpp"
#include <chrono>
#include <random>
#include <iomanip>

float CompareBias(string path, string name, string path2, string name2) {
#if !LINUX
    wchar_t szCmdLineW[500] = { 0 };
    char buffer[500] = { 0 };
    DWORD bytesRead = 0;
    string str = "cmd.exe /c python \"./tools/Compare.py\" \"";
    str += path + "\" \"" + name + "\" \"" + path2 + "\" \"" + name2 + "\"";
    size_t convertedChars = 0;
    mbstowcs_s(&convertedChars, szCmdLineW, str.length() + 1, str.c_str(), _TRUNCATE);
    SECURITY_ATTRIBUTES sa = { 0 };
    HANDLE hRead = NULL, hWrite = NULL;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.lpSecurityDescriptor = NULL;
    sa.bInheritHandle = TRUE;
    if (!CreatePipe(&hRead, &hWrite, &sa, 0))
        return -1;
    STARTUPINFOW si = { 0 };
    PROCESS_INFORMATION pi = { 0 };
    si.cb = sizeof(STARTUPINFO);
    GetStartupInfoW(&si);
    si.hStdError = hWrite;
    si.hStdOutput = hWrite;
    si.wShowWindow = SW_HIDE;
    si.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    if (!CreateProcessW(NULL, szCmdLineW, NULL, NULL, TRUE, NULL, NULL, NULL, &si, &pi)) {
        CloseHandle(hWrite);
        CloseHandle(hRead);
        return -1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(hWrite);
    ReadFile(hRead, buffer, 500, &bytesRead, NULL);
    CloseHandle(hRead);
    float bias = atof(buffer);
    return bias;
#else
    reutrn 0;
#endif
}

int main()
{
    string GT_path = "./Results/GT/";
    string predict_path = "./Results/";
    string test_path = "./TestCase/";

    int sample_num = 1024;
    VolumeRender::RenderType RunType = VolumeRender::RenderType::MRPNN;

    string Log_path = RunType == VolumeRender::RenderType::RPNN ? "./Log_RPNN.txt" : "./Log_MRPNN.txt";

    predict_path += RunType == VolumeRender::RenderType::RPNN ? "RPNN/" : "MRPNN/";

    const int test_num = 4;
    string names[test_num] = { "CLOUD0", "CLOUD1",  "MODEL0", "MODEL1"};
    float3 lightDirs[test_num][3] = { 
        {
            normalize(float3{ 0.34281, 0.70711, 0.61845 }), 
            normalize(float3{ 0.98528, 0.06976, 0.15605 }), 
            normalize(float3{ -0.90329, 0.34202, 0.55901}),
        },
        {
            normalize(float3{ -0.11803, 0.52992, 0.83979 }),
            normalize(float3{ 0.55721, 0.70711, 0.43534 }),
            normalize(float3{ -0.94026, -0.20791, -0.26961}),
        },
        {
            normalize(float3{ 0.20489, 0.87462, -0.43939 }),
            normalize(float3{ 1, 0, 0 }),
            normalize(float3{ -1,0,0 }),
        },
        {
            normalize(float3{ 0.98515, 0.12187, 0.12096 }),
            normalize(float3{ 0.39785, 0.20791, 0.89358 }),
            normalize(float3{ 0.05154, -0.17365, -0.98346 }),
        }
    };
    float3 CamPoss[test_num] = {
        float3{ 0.67085, -0.03808, -0.04856 }, float3{ 0.55471, 0.30303, 0.10048 }, 
        float3{ 0.8, 0, 0 }, float3{ -0.01786, 0.11180, 0.56642 }
    };
    float alphas[test_num] = { 1,2,2,4 };

    FILE* log = fopen(Log_path.c_str(), "w");
    float2 avrg = { 0 };
    int start_time = clock();
    for (int i = 0; i < test_num; i++)
    {
        string name = names[i];
        VolumeRender v(test_path + name);
        float3 lightColor = { 1.0, 1.0, 1.0 };
        float alpha = alphas[i];
        float3 CamPos = CamPoss[i];

        Camera cam(v, "test camera");
        cam.resolution = 512;
        cam.SetPosition(CamPos);

        string dirName[3] = { "side", "front", "back" };
        for (int j = 0; j < 3; j++)
        {
            float3 lightDir = lightDirs[i][j];

            #define RunTest(G, A0, A1, A2, Type) {\
                float g = G;\
                float3 scatter = float3{ A0, A1, A2 };\
                string filename = name + "." + #G + ".(" + #A0 + "," + #A1 + "," + #A2 + ")." + dirName[j];\
                v.SetScatterRate(scatter);\
                cam.RenderToFile(predict_path + name + "/" + filename, lightDir, lightColor, alpha, 512, g, sample_num, Camera::ACES, Type);\
                float bias = CompareBias(GT_path + name, filename, predict_path + name, filename);\
                printf("%s  %f\n", filename.c_str(), bias);\
                avrg.x += bias; avrg.y += 1;\
                fprintf(log, "%s  %f\n", filename.c_str(), bias);\
                fflush(log);\
            }

            RunTest(0.857, 1, 1, 1, RunType);
            if (RunType == VolumeRender::RenderType::MRPNN) {
                RunTest(0.5, 1, 1, 1, RunType);
                RunTest(0, 1, 1, 1, RunType);
                RunTest(0.857, 0.96, 0.98, 1, RunType);
                RunTest(0.857, 0.8, 0.9, 1, RunType);
            }
        }
    }

    printf("All Test done in  %.2f mins.\n", (clock() - start_time) / 1000.0f / 60);
    fprintf(log, "All Test done in  %.2f mins.\n", (clock() - start_time) / 1000.0f / 60);
    printf("Average bias:  %f", avrg.x / avrg.y);
    fprintf(log, "Average bias:  %f", avrg.x / avrg.y);
    fclose(log);

    return 0;
}