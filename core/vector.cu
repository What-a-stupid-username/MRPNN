#include "vector.cuh"

__device__ __host__ const float3 operator*(const float3 a, const float b) { return { a.x * b,a.y * b,a.z * b }; }
__device__ __host__ const float3 operator*(const float3 a, const float3 b) { return { a.x * b.x,a.y * b.y,a.z * b.z }; }

__device__ __host__ const float3 operator/(const float3 a, const float b) { return { a.x / b,a.y / b,a.z / b }; }
__device__ __host__ const float3 operator/(const float3 a, const float3 b) { return { a.x / b.x,a.y / b.y,a.z / b.z }; }

__device__ __host__ const float3 operator+(const float3 a, const float b) { return { a.x + b,a.y + b,a.z + b }; }
__device__ __host__ const float3 operator+(const float3 a, const float3 b) { return { a.x + b.x,a.y + b.y,a.z + b.z }; }

__device__ __host__ const float3 operator-(const float3 a, const float b) { return { a.x - b,a.y - b,a.z - b }; }
__device__ __host__ const float3 operator-(const float3 a, const float3 b) { return { a.x - b.x,a.y - b.y,a.z - b.z }; }
__device__ __host__ const float3 operator-(const float3 a) { return { -a.x, -a.y, -a.z }; }

__device__ __host__ const float4 make_float4(const float3 a, const float b) { return float4{ a.x, a.y, a.z, b }; }
__device__ __host__ const float3 make_float3(const float4 a) { return float3{ a.x, a.y, a.z }; }

__device__ __host__ const float3 inv(const float3 a) { return { a.x == 0 ? 999999999 : 1 / a.x , a.y == 0 ? 999999999 : 1 / a.y , a.z == 0 ? 999999999 : 1 / a.z }; }
__device__ __host__ float frac(const float f) { return f - (long)f; }
__device__ __host__ const float3 frac(const float3 f) { return { frac(f.x), frac(f.y), frac(f.z) }; }
__device__ __host__ float dot(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __host__ float saturate_(const float v) { return min(max(v, 0.0f), 1.0f); }
__device__ const float3 saturate(const float3 v) { return { saturate_(v.x),saturate_(v.y),saturate_(v.z) }; }
__device__ __host__ const float3 saturate_(const float3 v) { return { saturate_(v.x),saturate_(v.y),saturate_(v.z) }; }
__device__ __host__ const float3 max(const float3 a, const float3 b) { return { max(a.x, b.x) , max(a.y, b.y), max(a.z, b.z) }; }
__device__ __host__ const float3 min(const float3 a, const float3 b) { return { min(a.x, b.x) , min(a.y, b.y), min(a.z, b.z) }; }
__device__ __host__ const float3 normalize(const float3 v) { float len = sqrt(dot(v, v)); return { v.x / len, v.y / len, v.z / len }; }
__device__ __host__ float lerp(const float a, const float b, const float v) { return a * (float(1) - v) + b * v; }
__device__ __host__ const float3 lerp(const float3 a, const float3 b, const float v){ return { lerp(a.x,b.x, v) ,lerp(a.y,b.y, v) ,lerp(a.z,b.z, v) }; }
__device__ __host__ const float3 cross(const float3 a, const float3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
__device__ __host__ const float3 pow(const float3 a, const float b) { return { pow(a.x, b),pow(a.y, b) ,pow(a.z, b) }; }
__device__ __host__ const float3 exp(const float3 a) { return { exp(a.x),exp(a.y) ,exp(a.z) }; };
__device__ __host__ float sign(const float a) { return a > 0 ? 1 : -1; }
__device__ __host__ const float3 sign(const float3 a) { return float3{ sign(a.x), sign(a.y),sign(a.z) }; }
__device__ __host__ const float3 abs(const float3 a) { return { abs(a.x), abs(a.y), abs(a.z) }; }
__device__ __host__ const float3 sin(const float3 a) { return float3{ sin(a.x),sin(a.y),sin(a.z) }; }
__device__ __host__ const int3 floor(const float3 a) { return { int(a.x), int(a.y), int(a.z) }; }
__device__ __host__ float length(const float3 a) { return sqrt(dot(a, a)); }
__device__ __host__ float distance(const float3 a, const float3 b) { return length(a - b); }
__device__ __host__ float RayBoxOffset(float3 p, float3 dir)
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

__device__ __host__ float RayBoxDistance(float3 p, float3 dir)
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

__device__ __constant__ float3 Spn0[8] = {
float3{0.000000,0.000000,0.000000},
float3{1.000000,0.000000,0.000000},
float3{0.666667,-0.549602,0.503481},
float3{0.333333,0.082426,-0.939199},
float3{0.000000,0.608439,0.793601},
float3{-0.333333,-0.928397,-0.164220},
float3{-0.666667,0.628898,-0.400053},
float3{-1.000000,-0.000000,0.000000},
};

__device__ __constant__ float3 Spn1[8] = {
float3{0.000000,1.000000,0.000000},float3{-0.516051,0.714286,0.472745},float3{0.078990,0.428571,-0.900048},float3{0.602198,0.142857,0.785461},
float3{-0.974614,-0.142857,-0.172395},float3{0.762340,-0.428571,-0.484938},float3{-0.181685,-0.714286,0.675860},float3{-0.000000,-1.000000,-0.000000},
};

__device__ __constant__ float3 Spn2[16] = {
float3{0.000000,0.000000,1.000000},float3{0.336994,-0.367864,0.866667},float3{-0.677266,0.059438,0.733333},float3{0.634881,0.486751,0.600000},
float3{-0.154052,-0.870913,0.466667},float3{-0.506032,0.795500,0.333333},float3{0.946204,-0.254359,0.200000},float3{-0.885474,-0.459882,0.066667},
float3{0.342275,0.937232,-0.066667},float3{0.373847,-0.905670,-0.200000},float3{-0.853934,0.399606,-0.333333},float3{0.843895,0.264697,-0.466667},
float3{-0.401126,-0.692169,-0.600000},float3{-0.145981,0.664012,-0.733333},float3{0.408121,-0.286925,-0.866667},float3{-0.000000,-0.000000,-1.000000},
};

__device__ __constant__ float3 Spn3[16] = {
float3{0.628945,0.674911,-0.385907},float3{0.418351,0.439884,-0.794660},float3{0.926019,0.329212,0.184683},float3{-0.225970,0.925107,-0.305147},
float3{0.672535,-0.317517,-0.668490},float3{0.324828,0.606254,0.725908},float3{-0.470630,0.251300,-0.845787},float3{0.815736,-0.533206,0.224201},
float3{-0.575729,0.689246,0.439859},float3{-0.112624,-0.630502,-0.767974},float3{0.276062,-0.215587,0.936649},float3{-0.977336,0.119351,-0.174841},
float3{0.122688,-0.992256,0.019382},float3{-0.562275,-0.092724,0.821736},float3{-0.747574,-0.653927,-0.116242},float3{-0.628945,-0.674911,0.385907},
};

__device__ __constant__ float3 Spn4[16] = {
float3{-0.513486,-0.802627,-0.303516},float3{-0.329417,-0.930220,0.161785},float3{-0.113287,-0.521353,-0.845788},float3{-0.950328,-0.222046,0.218111},
float3{0.464981,-0.885166,0.016571},float3{-0.521976,0.229310,-0.821558},float3{-0.345623,-0.356336,0.868084},float3{0.701860,-0.285260,-0.652701},
float3{-0.803928,0.594684,0.007127},float3{0.593662,-0.395610,0.700755},float3{0.266437,0.545373,-0.794719},float3{-0.335359,0.492208,0.803284},
float3{0.994251,0.088891,0.059699},float3{-0.041402,0.990302,-0.132615},float3{0.469425,0.505778,0.723761},float3{0.513486,0.802627,0.303516},
};

//0.662213
__device__ __constant__ float3 SpVn5[32] = {
float3{0.382081,0.918886,-0.098303},float3{0.650970,0.595892,0.470267},
float3{0.550136,-0.754869,0.357104},float3{-0.181620,-0.226392,0.315110},
float3{0.896833,0.419476,-0.138502},float3{-0.846077,0.475248,0.241414},
float3{-0.118985,-0.729624,-0.673418},float3{-0.202309,0.321186,-0.001336},
float3{-0.387992,0.920914,0.037136},float3{-0.651299,-0.241163,-0.719479},
float3{-0.226884,0.258907,-0.938876},float3{0.217251,0.314177,0.919230},
float3{0.823361,-0.030952,0.566674},float3{-0.648360,-0.662882,0.374455},
float3{0.292507,0.335440,-0.491240},float3{-0.256196,-0.179743,0.949766},
float3{0.353981,-0.354754,0.865359},float3{0.543675,-0.709539,-0.448299},
float3{0.248602,-0.236637,-0.939250},float3{-0.072767,-0.771426,0.632144},
float3{0.376908,0.055425,0.107197},float3{-0.152183,0.820610,-0.550854},
float3{-0.762858,0.434625,-0.478684},float3{-0.604668,-0.753714,-0.257473},
float3{0.800153,-0.074823,-0.595110},float3{0.047339,0.850153,0.524403},
float3{-0.789310,-0.083667,0.608267},float3{-0.435870,0.459988,0.773575},
float3{0.940615,-0.339424,-0.005972},float3{-0.939158,-0.149618,-0.079873},
float3{0.017382,-0.945689,-0.034440},float3{-0.045413,-0.211751,-0.306299}
};

//0.658209
__device__ __constant__ float3 SpVn6[32] = {
float3{-0.727243,-0.193675,0.567283},float3{-0.295083,-0.828711,0.475567},
float3{-0.995703,-0.076032,-0.052861},float3{0.406178,0.627644,0.664141},
float3{0.453177,-0.018639,0.891226},float3{0.034613,0.066587,0.389293},
float3{-0.211494,0.858032,0.468029},float3{-0.805961,0.499459,0.317753},
float3{-0.711991,-0.228635,-0.654137},float3{0.720579,0.535021,-0.441042},
float3{0.373046,-0.669333,0.642518},float3{0.683159,-0.730082,0.016564},
float3{-0.386891,0.323358,0.863571},float3{0.948147,-0.134710,-0.287874},
float3{-0.129105,0.402723,-0.050831},float3{0.437607,-0.645013,-0.626465},
float3{0.876379,-0.234138,0.420880},float3{0.881941,0.419503,0.214937},
float3{-0.295534,-0.763040,-0.574830},float3{0.201295,-0.357166,0.034116},
float3{0.090924,-0.992554,-0.081048},float3{-0.329153,-0.143500,-0.075175},
float3{0.110731,0.736193,-0.667652},float3{0.527626,0.025130,-0.849105},
float3{-0.721017,-0.690451,-0.058410},float3{-0.161251,-0.320616,0.933383},
float3{-0.046937,-0.277278,-0.959643},float3{0.344410,0.938620,0.019314},
float3{0.127436,0.018955,-0.397832},float3{-0.793211,0.464290,-0.394019},
float3{-0.358985,0.309343,-0.880589},float3{-0.337826,0.918535,-0.205349}
};

//0.644014
__device__ __constant__ float3 SpVn7[32] = {
float3{-0.410460,-0.591758,-0.693790},float3{0.345327,0.801148,0.488785},
float3{-0.331667,-0.940133,-0.075232},float3{-0.862656,-0.487171,-0.126874},
float3{0.369731,0.254916,0.014690},float3{-0.744325,0.659225,0.106789},
float3{-0.053761,-0.828511,0.557386},float3{-0.125478,-0.240730,0.962237},
float3{0.062130,-0.197417,-0.351812},float3{-0.657577,-0.492287,0.570303},
float3{0.353328,0.715205,-0.602790},float3{-0.383008,-0.047040,0.228408},
float3{0.816554,-0.436954,-0.377240},float3{0.557326,-0.800918,0.218903},
float3{-0.818517,0.251103,-0.505864},float3{0.829851,0.427577,0.358505},
float3{-0.194640,0.403027,-0.118038},float3{0.177782,-0.232706,0.295433},
float3{-0.253527,0.068370,-0.947231},float3{0.017371,0.305274,0.491131},
float3{0.934119,-0.225524,0.276239},float3{-0.260850,0.723730,-0.638884},
float3{0.264240,-0.861446,-0.433692},float3{-0.555007,0.258905,0.790529},
float3{0.597648,0.800227,-0.049532},float3{0.510981,0.224669,0.829338},
float3{-0.250464,0.888485,0.384352},float3{0.513797,-0.376265,0.770998},
float3{0.539898,0.053152,-0.840034},float3{0.947216,0.229984,-0.223358},
float3{-0.966975,0.104162,0.232617},float3{0.021570,0.986682,-0.159492}
};

__device__ __host__ const float3 Roberts2(const int n) {
    const float g = 1.32471795724474602596;
    const float3 a = float3{ 1.0f / g, 1.0f / (g * g) };
    return  frac(a * n + 0.5);
}

__device__ __host__ const float3 UniformSampleHemisphere(const float x, const float y) {
    float Phi = float(2 * 3.14159265359 * x);
    float CosTheta = y;
    float SinTheta = sqrt(1 - CosTheta * CosTheta);

    float3 H = { SinTheta * cos(Phi), SinTheta * sin(Phi), CosTheta };

    return H;
}

__device__ __host__ float3 UniformSampleSphere(float2 E) {
    float Phi = 2 * 3.14159265359f * E.x;
    float CosTheta = 1 - 2 * E.y;
    float SinTheta = sqrt(1 - CosTheta * CosTheta);

    float3 H;
    H.x = SinTheta * cos(Phi);
    H.y = SinTheta * sin(Phi);
    H.z = CosTheta;

    //float PDF = 1 / (4 * 3.14159265359f);

    return H;
}

__device__ __host__ float HenyeyGreenstein(float cos, float g)
{
    float g2 = g * g;
    return ((1.0f - g2) / pow((1.0f + g2 - 2.0f * g * cos), 1.5f)) * 0.25f /*/ 3.14159265359*/;
}

__device__ __host__ float SampleHeneyGreenstein(float s, float g) {
    if (abs(g) < 0.0001) return s * 2.0 - 1.0;
    float g2 = g * g;
    float t0 = (1 - g2) / (1 - g + 2 * g * s);
    float cosAng = (1 + g2 - t0 * t0) / (2 * g);

    return cosAng;
}

__device__ __host__ float3 SampleHenyeyGreenstein(const float e0, const float e1, const float3 v, const float g) {
    float CosTheta = SampleHeneyGreenstein(e0, g);

    float Phi = 2 * 3.14159265359 * e1;
    float SinTheta = sqrt(max(0.0f, 1.0f - CosTheta * CosTheta));

    float3 v2, v3;

    if (abs(v.z) > sqrt(0.5)) {
        v2 = normalize(cross(v, { 0, 1, 0 }));
    }
    else {
        v2 = normalize(cross(v, { 0, 0, 1 }));
    }
    v3 = cross(v2, v);

    return (v3 * (SinTheta * cos(Phi))) + (v2 * (SinTheta * sin(Phi))) + (v * CosTheta);
}

__device__ __host__ float3 SampleHenyeyGreenstein_HG(const float e0, const float e1, const float3 v, const float g, float& hg) {
    float CosTheta = SampleHeneyGreenstein(e0, g);
    float Phi = 2 * 3.14159265359 * e1;
    float SinTheta = sqrt(max(0.0f, 1.0f - CosTheta * CosTheta));
    float3 v2, v3;
    if (abs(v.z) > sqrt(0.5)) {
        v2 = normalize(cross(v, { 0, 1, 0 }));
    }
    else {
        v2 = normalize(cross(v, { 0, 0, 1 }));
    }
    v3 = cross(v2, v);
    float g2 = g * g;
    hg = ((1.0f - g2) / pow((1.0f + g2 - 2.0f * g * CosTheta), 1.5f)) * 0.25f / 3.14159265359;//;// 
    return (v3 * (SinTheta * cos(Phi))) + (v2 * (SinTheta * sin(Phi))) + (v * CosTheta);
}

__device__ __host__ const float3 float3x3::operator*(const float3 v) const {
    return { dot(x, v), dot(y, v), dot(z, v) };
}

__device__ const float3 SphereRandom3(int index, float radius, float3 XMain, float3 YMain, float3 ZMain, float g)
{
    float3 LocalFramePos;
    if (index < 8)
    {
        LocalFramePos = Spn0[index];
    }
    else if (index < 16)
    {
        LocalFramePos = Spn1[index - 8];
    }
    else if (index < 32)
    {
        LocalFramePos = Spn2[index - 16];
    }
    else if (index < 48)
    {
        LocalFramePos = Spn3[index - 32];
    }
    else if (index < 64)
    {
        LocalFramePos = Spn4[index - 48];
    }
    else if (index < 96)
    {
        LocalFramePos = SpVn5[index - 64];
    }
    else if (index < 128)
    {
        LocalFramePos = SpVn6[index - 96];
    }
    else if (index < 160)
    {
        LocalFramePos = SpVn7[index - 128];
    }
    //float e1 = RandomSequence[index + 64];
    return (XMain * LocalFramePos.x + YMain * LocalFramePos.y + ZMain * LocalFramePos.z) * radius;// *lerp(0.5f, 1.0f, e1);
}

__device__ __host__ const Offset_Layer_ GetSamples23_(int index)
{
    const float MsOffsetScale = 0.6;
    //sp0
    if (index < 8)
    {
        float currentMip = 0.0;
        float offset = 1.0 / 256.0;//1.0
        offset *= lerp(MsOffsetScale, 1.0, 1.0);
        return Offset_Layer_(offset, currentMip, 5,index);
    }
    //sp1
    else if (index < 16)
    {
        float currentMip = 1.0;
        float offset = 2.5 / 256.0;//2.5
        offset *= lerp(MsOffsetScale, 1.0, 0.8);
        return Offset_Layer_(offset, currentMip, 5,8);
    }
    //sp2
    else if (index < 32)
    {
        float currentMip = 2.0;
        float offset = 5.5 / 256.0;//5.5
        offset *= lerp(MsOffsetScale, 1.0, 0.6);
        return Offset_Layer_(offset, currentMip, 5,8);
    }
    //sp3
    else if (index < 48)
    {
        float currentMip = 3.0;//radius 4
        float offset = 11.5 / 256.0;
        offset *= lerp(MsOffsetScale, 1.0, 0.4);
        return Offset_Layer_(offset, currentMip, 5,8);
    }
    //sp4
    else if (index < 64)
    {
        float currentMip = 4.0;//radius 8
        float offset = 23.5 / 256.0;
        offset *= lerp(MsOffsetScale, 1.0, 0.2);
        return Offset_Layer_(offset, currentMip, 5,8);
    }
    //sp5
    else if (index < 96)
    {
        float currentMip = 5.0;//radius 16
        float offset = 47.5 / 256.0;
        offset *= MsOffsetScale;
        return Offset_Layer_(offset, currentMip, 5,8);
    }
    //sp6
    else if (index < 128)
    {
        float currentMip = 6.0;//radius 32
        float offset = 95.5 / 256.0;
        offset *= MsOffsetScale;
        return Offset_Layer_(offset, currentMip, 5,8);
    }
    //sp7
    else if (index < 160)
    {
        float currentMip = 7.0;//radius 64
        float offset = 191.5 / 256.0;
        offset *= MsOffsetScale;
        return Offset_Layer_(offset, currentMip, 5, 8);
    }
    //light dir
    else if (index < 168)
    {
        int i = index - 160;
        float currentMip = 0.0;
        float offset = i + 1;//1,2,3,4,5,6,7,8
        offset /= 256.0;
        return Offset_Layer_(offset, currentMip, 1, 8);
    }
    else if (index < 176)
    {
        int i = index - 168;
        float currentMip = 1.0;
        float offset = 8.5 + 1 + 2.0 * i;//9.5,11.5,13.5,15.5 ,17.5,19.5,21.5,23.5
        offset /= 256.0;
        return Offset_Layer_(offset, currentMip, 1, 8);
    }
    else if (index < 184)
    {
        int i = index - 176;
        float currentMip = 2.0;
        float offset = 26.5 + 4.0 * i;//26.5,30.5,34.5,38.5 , 42.5,46.5,50.5,54.5
        offset /= 256.0;
        return Offset_Layer_(offset, currentMip, 1, 8);
    }
    else if (index < 192)
    {
        int i = index - 184;
        float currentMip = 3.0;
        float offset = 60.5 + 8.0 * i;//60.5,68.5,76.5,84.5, 92.5,100.5,108.5,116.5
        offset /= 256.0;
        return Offset_Layer_(offset, currentMip, 1, 8);
    }
    return Offset_Layer_(0.0f, 0.0, 1, 8);
}


__device__ float3 HashALU(float3 p, float numCells)
{
    // This is tiling part, adjusts with the scale
    //p = mod(p, numCells);

    p = float3{ dot(p, float3{ 127.1, 311.7, 74.7 }),
        dot(p, float3{269.5, 183.3, 246.1}),
        dot(p, float3{113.5, 271.9, 124.6}) };

    return frac(sin(p) * 43758.5453123) * 2.0 - 1.0;
}

#define Hash HashALU

__device__ float TileableNoise(float3 p, float numCells)
{
    float3 f, i;

    f = frac(p);		// Separate integer from fractional
    int3 ii = floor(p);
    i = float3{ (float)ii.x, (float)ii.y, (float)ii.z };

    float3 u = f * f * (f * -2.0 + 3.0); // Cosine interpolation approximation

    return lerp(lerp(lerp(dot(Hash(i + float3{ 0.0, 0.0, 0.0 }, numCells), f - float3{ 0.0, 0.0, 0.0 }),
        dot(Hash(i + float3{ 1.0, 0.0, 0.0 }, numCells), f - float3{ 1.0, 0.0, 0.0 }), u.x),
        lerp(dot(Hash(i + float3{ 0.0, 1.0, 0.0 }, numCells), f - float3{ 0.0, 1.0, 0.0 }),
            dot(Hash(i + float3{ 1.0, 1.0, 0.0 }, numCells), f - float3{ 1.0, 1.0, 0.0 }), u.x), u.y),
        lerp(lerp(dot(Hash(i + float3{ 0.0, 0.0, 1.0 }, numCells), f - float3{ 0.0, 0.0, 1.0 }),
            dot(Hash(i + float3{ 1.0, 0.0, 1.0 }, numCells), f - float3{ 1.0, 0.0, 1.0 }), u.x),
            lerp(dot(Hash(i + float3{ 0.0, 1.0, 1.0 }, numCells), f - float3{ 0.0, 1.0, 1.0 }),
                dot(Hash(i + float3{ 1.0, 1.0, 1.0 }, numCells), f - float3{ 1.0, 1.0, 1.0 }), u.x), u.y), u.z);
}

__device__ float TileableNoiseFBM(float3 p, float numCells, int octaves)
{
    float f = 0.0;

    // Change starting scale to any integer value...
    //p = mod(p, float3(numCells));
    float amp = 0.5;
    float sum = 0.0;

    for (int i = 0; i < octaves; i++)
    {
        f += TileableNoise(p, numCells) * amp;
        sum += amp;
        amp *= 0.5;

        // numCells must be multiplied by an integer value...
        numCells *= 2.0;
    }

    return f / sum;
}

__device__ float3 snoiseVec3(float3 x, float numCells, int octaves)
{

    float s = TileableNoiseFBM(x, numCells, octaves);
    float s1 = TileableNoiseFBM(float3{ x.y - 19.1f, x.z + 33.4f, x.x + 47.2f }, numCells, octaves);
    float s2 = TileableNoiseFBM(float3{ x.z + 74.2f, x.x - 124.5f, x.y + 99.4f }, numCells, octaves);
    float3 c = float3{ s, s1, s2 };
    return c;

}

__device__ float3 TileableCurlNoise(float3 p, float numCells, int octaves)
{
    const float e = .1;
    float3 dx = float3{ e, 0.0, 0.0 };
    float3 dy = float3{ 0.0, e, 0.0 };
    float3 dz = float3{ 0.0, 0.0, e };

    float3 p_x0 = snoiseVec3(p - dx, numCells, octaves);
    float3 p_x1 = snoiseVec3(p + dx, numCells, octaves);
    float3 p_y0 = snoiseVec3(p - dy, numCells, octaves);
    float3 p_y1 = snoiseVec3(p + dy, numCells, octaves);
    float3 p_z0 = snoiseVec3(p - dz, numCells, octaves);
    float3 p_z1 = snoiseVec3(p + dz, numCells, octaves);

    float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
    float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
    float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

    const float divisor = 1.0 / (2.0 * e);
    return normalize(float3{ x, y, z } *divisor);
}
