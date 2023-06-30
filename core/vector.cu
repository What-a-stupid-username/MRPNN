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
__device__ __host__ const float frac(const float f) { return f - (long)f; }
__device__ __host__ const float3 frac(const float3 f) { return { frac(f.x), frac(f.y), frac(f.z) }; }
__device__ __host__ const float dot(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __host__ const float saturate_(const float v) { return min(max(v, 0.0f), 1.0f); }
__device__ const float3 saturate(const float3 v) { return { saturate_(v.x),saturate_(v.y),saturate_(v.z) }; }
__device__ __host__ const float3 saturate_(const float3 v) { return { saturate_(v.x),saturate_(v.y),saturate_(v.z) }; }
__device__ __host__ const float3 max(const float3 a, const float3 b) { return { max(a.x, b.x) , max(a.y, b.y), max(a.z, b.z) }; }
__device__ __host__ const float3 min(const float3 a, const float3 b) { return { min(a.x, b.x) , min(a.y, b.y), min(a.z, b.z) }; }
__device__ __host__ const float3 normalize(const float3 v) { float len = sqrt(dot(v, v)); return { v.x / len, v.y / len, v.z / len }; }
__device__ __host__ const float lerp(const float a, const float b, const float v) { return a * (float(1) - v) + b * v; }
__device__ __host__ const float3 lerp(const float3 a, const float3 b, const float v){ return { lerp(a.x,b.x, v) ,lerp(a.y,b.y, v) ,lerp(a.z,b.z, v) }; }
__device__ __host__ const float3 cross(const float3 a, const float3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
__device__ __host__ const float3 pow(const float3 a, const float b) { return { pow(a.x, b),pow(a.y, b) ,pow(a.z, b) }; }
__device__ __host__ const float3 exp(const float3 a) { return { exp(a.x),exp(a.y) ,exp(a.z) }; };
__device__ __host__ const float sign(const float a) { return a > 0 ? 1 : -1; }
__device__ __host__ const float3 sign(const float3 a) { return float3{ sign(a.x), sign(a.y),sign(a.z) }; }
__device__ __host__ const float3 abs(const float3 a) { return { abs(a.x), abs(a.y), abs(a.z) }; }
__device__ __host__ const float3 sin(const float3 a) { return float3{ sin(a.x),sin(a.y),sin(a.z) }; }
__device__ __host__ const int3 floor(const float3 a) { return { int(a.x), int(a.y), int(a.z) }; }
__device__ __host__ const float length(const float3 a) { return sqrt(dot(a, a)); }
__device__ __host__ const float distance(const float3 a, const float3 b) { return length(a - b); }
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


__device__ const float3 SphereX[64] = {
float3{-0.346710f,-0.893468f,0.285495f},float3{-0.287419f,-0.807652f,0.514868f},float3{-0.157235f,-0.987560f,-0.001263f},float3{-0.695936f,-0.636101f,0.333238f},
float3{0.109879f,-0.861153f,0.496329f},float3{-0.480938f,-0.837667f,-0.258867f},float3{-0.474211f,-0.487844f,0.732893f},float3{0.255954f,-0.966637f,0.010019f},
float3{-0.860367f,-0.509068f,-0.024877f},float3{0.099692f,-0.579736f,0.808683f},float3{-0.111179f,-0.872697f,-0.475435f},float3{-0.794677f,-0.242149f,0.556643f},
float3{0.518581f,-0.788225f,0.331322f},float3{-0.711382f,-0.529093f,-0.462597f},float3{-0.199623f,-0.240833f,0.949816f},float3{0.363377f,-0.849498f,-0.382506f},
float3{-0.984646f,-0.132226f,0.113969f},float3{0.496648f,-0.476016f,0.725775f},float3{-0.288700f,-0.604802f,-0.742204f},float3{-0.606192f,0.044820f,0.794054f},
float3{0.718718f,-0.694916f,-0.023184f},float3{-0.896673f,-0.157345f,-0.413788f},float3{0.194657f,-0.101337f,0.975623f},float3{0.258543f,-0.634963f,-0.727995f},
float3{-0.911108f,0.201891f,0.359336f},float3{0.801672f,-0.400142f,0.444082f},float3{-0.516421f,-0.256128f,-0.817134f},float3{-0.257003f,0.239046f,0.936380f},
float3{0.728831f,-0.541200f,-0.419414f},float3{-0.953660f,0.213491f,-0.212023f},float3{0.580817f,-0.018638f,0.813821f},float3{0.041664f,-0.334740f,-0.941389f},
float3{-0.661986f,0.462989f,0.589420f},float3{0.952769f,-0.298425f,0.056333f},float3{-0.684961f,0.125288f,-0.717727f},float3{0.154520f,0.358570f,0.920625f},
float3{0.586566f,-0.307182f,-0.749386f},float3{-0.842293f,0.537037f,0.046203f},float3{0.863726f,0.055827f,0.500861f},float3{-0.186526f,0.025040f,-0.982131f},
float3{-0.292607f,0.642006f,0.708667f},float3{0.932599f,-0.131718f,-0.336021f},float3{-0.713846f,0.490313f,-0.500016f},float3{0.525869f,0.437602f,0.729360f},
float3{0.364504f,0.005148f,-0.931188f},float3{-0.568325f,0.781989f,0.255930f},float3{0.978954f,0.170941f,0.111478f},float3{-0.323559f,0.402692f,-0.856241f},
float3{0.108490f,0.751941f,0.650242f},float3{0.769852f,0.122941f,-0.626269f},float3{-0.555535f,0.791023f,-0.256248f},float3{0.755758f,0.524761f,0.391735f},
float3{0.166289f,0.372066f,-0.913189f},float3{-0.182681f,0.933452f,0.308700f},float3{0.894523f,0.374186f,-0.244568f},float3{-0.266977f,0.742142f,-0.614775f},
float3{0.420375f,0.826809f,0.373728f},float3{0.536483f,0.474970f,-0.697560f},float3{-0.185347f,0.975419f,-0.119182f},float3{0.724797f,0.688587f,-0.022736f},
float3{0.145851f,0.749829f,-0.645355f},float3{0.224975f,0.973555f,0.039699f},float3{0.539045f,0.755629f,-0.372096f},float3{0.346710f,0.893468f,-0.285495f},
};
__device__ const float3 Sp0[8] = {
float3{1.000000f,0.000000f,0.000000f},
float3{0.714286f,-0.516051f,0.472745f},
float3{0.428571f,0.078990f,-0.900048f},
float3{0.142857f,0.602198f,0.785461f},
float3{-0.142857f,-0.974614f,-0.172395f},
float3{-0.428571f,0.762340f,-0.484938f},
float3{-0.714286f,-0.181685f,0.675860f},
float3{-1.000000f,-0.000000f,-0.000000f},
};

__device__ const float3 Sp1[8] = {
float3{0.000000,1.000000f,0.000000f},
float3{-0.516051f,0.714286f,0.472745f},
float3{0.078990f,0.428571f,-0.900048f},
float3{0.602198f,0.142857f,0.785461f},
float3{-0.974614f,-0.142857f,-0.172395f},
float3{0.762340f,-0.428571f,-0.484938f},
float3{-0.181685f,-0.714286f,0.675860f},
float3{-0.000000f,-1.000000f,-0.000000f},
};

__device__ const float3 Sp2[8] = {
float3{0.000000f,0.000000,1.000000f},
float3{0.472745f,-0.516051f,0.714286f},
float3{-0.900048f,0.078990f,0.428571f},
float3{0.785461f,0.602198f,0.142857f},
float3{-0.172395f,-0.974614f,-0.142857f},
float3{-0.484938f,0.762340f,-0.428571f},
float3{0.675860f,-0.181685f,-0.714286f},
float3{-0.000000f,-0.000000f,-1.000000f},
};

__device__ const float3 Sp3[16] = {
float3{0.628945f,0.674911f,-0.385907f},
float3{0.418351f,0.439884f,-0.794660f},
float3{0.926019f,0.329212f,0.184683f},
float3{-0.225970f,0.925107f,-0.305147f},
float3{0.672535f,-0.317517f,-0.668490f},
float3{0.324828f,0.606254f,0.725908f},
float3{-0.470630f,0.251300f,-0.845787f},
float3{0.815736f,-0.533206f,0.224201f},
float3{-0.575729f,0.689246f,0.439859f},
float3{-0.112624f,-0.630502f,-0.767974f},
float3{0.276062f,-0.215587f,0.936649f},
float3{-0.977336f,0.119351f,-0.174841f},
float3{0.122688f,-0.992256f,0.019382f},
float3{-0.562275f,-0.092724f,0.821736f},
float3{-0.747574f,-0.653927f,-0.116242f},
float3{-0.628945f,-0.674911f,0.385907f},
};

__device__ const float3 Sp4[16] = {
float3{-0.513486f,-0.802627f,-0.303516f},
float3{-0.329417f,-0.930220f,0.161785f},
float3{-0.113287f,-0.521353f,-0.845788f},
float3{-0.950328f,-0.222046f,0.218111f},
float3{0.464981f,-0.885166f,0.016571f},
float3{-0.521976f,0.229310f,-0.821558f},
float3{-0.345623f,-0.356336f,0.868084f},
float3{0.701860f,-0.285260f,-0.652701f},
float3{-0.803928f,0.594684f,0.007127f},
float3{0.593662f,-0.395610f,0.700755f},
float3{0.266437f,0.545373f,-0.794719f},
float3{-0.335359f,0.492208f,0.803284f},
float3{0.994251f,0.088891f,0.059699f},
float3{-0.041402f,0.990302f,-0.132615f},
float3{0.469425f,0.505778f,0.723761f},
float3{0.513486f,0.802627f,0.303516f},
};

__device__ const float3 Sp5[16] = {
float3{0.711413f,-0.076100f,0.698642f},
float3{0.814789f,0.364848f,0.450560f},
float3{0.588713f,-0.717653f,0.372010f},
float3{-0.007315f,0.412035f,0.911139f},
float3{0.952671f,0.079255f,-0.293490f},
float3{-0.226951f,-0.745190f,0.627045f},
float3{0.165939f,0.960833f,0.221957f},
float3{0.503046f,-0.709119f,-0.494059f},
float3{-0.743435f,0.049393f,0.666982f},
float3{0.418242f,0.642397f,-0.642183f},
float3{-0.374894f,-0.906518f,-0.194112f},
float3{-0.647386f,0.758573f,0.073885f},
float3{0.110762f,-0.128135f,-0.985552f},
float3{-0.953036f,-0.282198f,-0.109937f},
float3{-0.485043f,0.540110f,-0.687760f},
float3{-0.711413f,0.076100f,-0.698642f},
};

__device__ const float3 Sp6[16] = {
float3{-0.346710f,-0.893468f,0.285495f},
float3{-0.204116f,-0.659676f,0.723301f},
float3{0.070437f,-0.947636f,-0.311488f},
float3{-0.925998f,-0.212498f,0.312044f},
float3{0.586386f,-0.564102f,0.581326f},
float3{-0.445634f,-0.447837f,-0.775147f},
float3{-0.392345f,0.214234f,0.894522f},
float3{0.811278f,-0.490903f,-0.317557f},
float3{-0.878466f,0.295537f,-0.375441f},
float3{0.555620f,0.260532f,0.789563f},
float3{0.277421f,-0.041285f,-0.959861f},
float3{-0.501061f,0.812209f,0.298756f},
float3{0.954911f,0.296570f,-0.013827f},
float3{-0.172263f,0.651550f,-0.738789f},
float3{0.296278f,0.927662f,0.227296f},
float3{0.346710f,0.893468f,-0.285495f},
};

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

__device__ const float3 Spn5[32] = {
float3{0.711413,-0.076100,0.698642},float3{0.805926,0.233952,0.543829},float3{0.668046,-0.544596,0.507080},float3{0.252820,0.276923,0.927036},
float3{0.998346,0.030542,0.048699},float3{0.119834,-0.613171,0.780808},float3{0.455105,0.740497,0.494514},float3{0.771978,-0.631786,-0.069974},
float3{-0.266230,0.002050,0.963907},float3{0.817684,0.549193,-0.172567},float3{0.115832,-0.951098,0.286350},float3{-0.134706,0.760234,0.635530},
float3{0.815296,-0.228817,-0.531916},float3{-0.511385,-0.502927,0.696814},float3{0.331224,0.938602,-0.096523},float3{0.268494,-0.906453,-0.325965},
float3{-0.648541,0.385683,0.656234},float3{0.605366,0.344788,-0.717393},float3{-0.482917,-0.859544,0.167256},float3{-0.284585,0.956039,0.070717},
float3{0.330000,-0.492628,-0.805244},float3{-0.906548,-0.132403,0.400799},float3{0.129336,0.748488,-0.650413},float3{-0.339947,-0.832680,-0.437128},
float3{-0.790110,0.606941,0.085731},float3{0.117474,0.044095,-0.992096},float3{-0.889982,-0.442145,-0.111534},float3{-0.448984,0.722236,-0.526106},
float3{-0.348245,-0.407682,-0.844110},float3{-0.953697,0.153174,-0.258844},float3{-0.445930,0.259342,-0.856673},float3{-0.711413,0.076100,-0.698642},
};

__device__ const float3 Spn6[32] = {
float3{-0.346710,-0.893468,0.285495},float3{-0.256084,-0.754607,0.604143},float3{-0.067320,-0.989518,-0.127761},float3{-0.810276,-0.481371,0.334268},
float3{0.309947,-0.774447,0.551510},float3{-0.492390,-0.722296,-0.485634},float3{-0.472986,-0.230733,0.850322},float3{0.509164,-0.851466,-0.125531},
float3{-0.958530,-0.225354,-0.174459},float3{0.305163,-0.298854,0.904191},float3{0.037472,-0.653310,-0.756163},float3{-0.817850,0.168262,0.550281},
float3{0.831204,-0.493407,0.256220},float3{-0.675059,-0.149421,-0.722474},float3{-0.041940,0.219422,0.974728},float3{0.628262,-0.476434,-0.615059},
float3{-0.923409,0.381764,-0.039649},float3{0.769301,0.013884,0.638735},float3{-0.097122,-0.098261,-0.990410},float3{-0.419051,0.625608,0.658035},
float3{0.969516,-0.119614,-0.213850},float3{-0.651547,0.456341,-0.606003},float3{0.430298,0.534758,0.727239},float3{0.487770,0.074354,-0.869800},
float3{-0.523245,0.846949,0.094296},float3{0.917654,0.373363,0.136059},float3{-0.105711,0.530451,-0.841098},float3{0.116690,0.911294,0.394875},
float3{0.736169,0.457600,-0.498656},float3{-0.154518,0.918362,-0.364330},float3{0.535658,0.844283,0.016021},float3{0.346710,0.893468,-0.285495},
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

__device__ const float3 Cir[32] = {
float3{1.000000f,0.000000f,0.000000f},
float3{-0.500000f,0.866025f,0.000000f},
float3{-0.500000f,-0.866025f,0.000000f},
float3{0.000000f,0.000000f,0.000000f},
float3{0.540302f,0.841471f,0.000000f},
float3{-0.998886f,0.047180f,0.000000f},
float3{0.458584f,-0.888651f,0.000000f},
float3{0.000000f,0.000000f,0.000000f},
float3{-0.416147f,0.909297f,0.000000f},
float3{-0.579401f,-0.815042f,0.000000f},
float3{0.995548f,-0.094255f,0.000000f},
float3{0.000000f,0.000000f,0.000000f},
float3{-0.989992f,0.141120f,0.000000f},
float3{0.372783f,-0.927919f,0.000000f},
float3{0.617210f,0.786799f,0.000000f},
float3{0.000000f,0.000000f,0.000000f},
float3{-0.653644f,-0.756802f,0.000000f},
float3{0.982232f,-0.187671f,0.000000f},
float3{-0.328588f,0.944473f,0.000000f},
float3{0.000000f,0.000000f,0.000000f},
float3{0.283662f,-0.958924f,0.000000f},
float3{0.688622f,0.725121f,0.000000f},
float3{-0.972284f,0.233803f,0.000000f},
float3{0.000000f,0.000000f,0.000000f},
float3{0.960170f,-0.279415f,0.000000f},
float3{-0.238104f,0.971240f,0.000000f},
float3{-0.722066f,-0.691824f,0.000000f},
float3{0.000000f,0.000000f,0.000000f},
float3{0.753902f,0.656987f,0.000000f},
float3{-0.945918f,0.324405f,0.000000f},
float3{0.192016f,-0.981392f,0.000000f},
float3{0.000000f,0.000000f,0.000000f},
};

__device__ const float RandomSequence[512] =
{
0.12796868f,0.04966642f,0.28448910f,0.44371054f,0.14441879f,0.27093321f,0.47752890f,0.90537047f,0.68565220f,0.07168046f,0.52776992f,0.47808480f,0.53827232f,0.33695608f,0.19808882f,0.01123386f,
0.47743893f,0.89680660f,0.65208942f,0.04214348f,0.87539071f,0.29477328f,0.27947390f,0.90935671f,0.06135495f,0.36617023f,0.48926046f,0.62545538f,0.73270398f,0.47092366f,0.00367865f,0.86067271f,
0.52059007f,0.28816774f,0.30438322f,0.66500884f,0.55910099f,0.78191215f,0.57037938f,0.24475318f,0.85359257f,0.09814926f,0.72283798f,0.39186493f,0.43510535f,0.92092681f,0.40309879f,0.91254431f,
0.81773347f,0.05518819f,0.95468777f,0.69312412f,0.34996146f,0.23416165f,0.60248083f,0.41131639f,0.60033190f,0.09174132f,0.03677176f,0.33303592f,0.56266499f,0.04045041f,0.19370860f,0.08325507f,
0.32861814f,0.49809185f,0.74826396f,0.88771909f,0.28000396f,0.31864330f,0.13247229f,0.13359657f,0.41679257f,0.85531026f,0.52546149f,0.85189790f,0.77623707f,0.92856032f,0.76444221f,0.59397054f,
0.98374850f,0.71912998f,0.28709468f,0.33370996f,0.95329159f,0.88957554f,0.74502635f,0.55362350f,0.98131686f,0.78179812f,0.88665944f,0.54398185f,0.82224852f,0.08036802f,0.62723690f,0.15086667f,
0.57845986f,0.37550083f,0.03858579f,0.85846382f,0.69414413f,0.17105809f,0.99206036f,0.11093671f,0.02636837f,0.51752192f,0.96283460f,0.80260545f,0.44608220f,0.72727686f,0.39657602f,0.42983070f,
0.44640681f,0.68367070f,0.76354063f,0.39969841f,0.57324618f,0.50856698f,0.95332193f,0.55456305f,0.29036510f,0.83998132f,0.09854488f,0.11261364f,0.92034936f,0.72578180f,0.26348031f,0.49880922f,
0.10128261f,0.30206612f,0.35727304f,0.79542673f,0.47312418f,0.34933344f,0.90636343f,0.49949256f,0.86685532f,0.86919808f,0.30209804f,0.31293750f,0.59647495f,0.69867402f,0.74276823f,0.04288172f,
0.38234472f,0.50630885f,0.44258013f,0.95559090f,0.01487583f,0.39590207f,0.51015395f,0.30524096f,0.23588340f,0.60869884f,0.41785458f,0.15623277f,0.33448061f,0.68133491f,0.65504199f,0.43576324f,
0.98340100f,0.01231503f,0.23118998f,0.45652521f,0.36164844f,0.13755344f,0.95601773f,0.22850378f,0.00675153f,0.25811577f,0.54144126f,0.60322642f,0.95678979f,0.28420949f,0.64610815f,0.33913451f,
0.79051834f,0.08868829f,0.29472545f,0.80539417f,0.48459035f,0.80487943f,0.11063510f,0.72047377f,0.41357827f,0.52848971f,0.87670654f,0.74805892f,0.20982459f,0.53174853f,0.18382213f,0.19322561f,
0.54406357f,0.41501209f,0.64975083f,0.90571201f,0.55256557f,0.60576856f,0.13421577f,0.55931705f,0.86388433f,0.67565703f,0.16254352f,0.82067418f,0.95986652f,0.80865169f,0.15980868f,0.75038487f,
0.89733994f,0.45453414f,0.55577904f,0.38193032f,0.25941354f,0.66641414f,0.10240409f,0.67299181f,0.19490381f,0.97911060f,0.42105070f,0.40472841f,0.51085913f,0.60487282f,0.59795403f,0.05492270f,
0.01988492f,0.24770483f,0.96063471f,0.57245046f,0.85347342f,0.09485047f,0.13176754f,0.71735775f,0.77050751f,0.29431105f,0.53803194f,0.73037410f,0.10296273f,0.69784063f,0.48075894f,0.00030270f,
0.15237473f,0.03653796f,0.38223302f,0.41178828f,0.70295209f,0.48463711f,0.08478009f,0.89785588f,0.46374774f,0.50583076f,0.30258429f,0.97460687f,0.11070360f,0.90053833f,0.02952959f,0.13058853f,
0.57302606f,0.98208165f,0.54384500f,0.67754912f,0.38774344f,0.06666636f,0.24828123f,0.15932254f,0.61346960f,0.83485967f,0.16758941f,0.14533350f,0.56750762f,0.84794539f,0.58779389f,0.60038978f,
0.38051379f,0.30082640f,0.31170321f,0.39991140f,0.25187358f,0.07071635f,0.50144672f,0.81654823f,0.84632814f,0.20085725f,0.18341739f,0.97794873f,0.44978413f,0.39043045f,0.42888623f,0.02281019f,
0.37251210f,0.97273123f,0.70035934f,0.76025552f,0.03939756f,0.94864053f,0.91957808f,0.65286720f,0.78350019f,0.08716749f,0.79820067f,0.35100782f,0.93511289f,0.38599458f,0.95139760f,0.31562665f,
0.68682098f,0.26310080f,0.71553808f,0.93869454f,0.33381715f,0.21698479f,0.75524276f,0.18014526f,0.41784203f,0.93866020f,0.15809397f,0.86762619f,0.32909063f,0.58698016f,0.89043635f,0.70160270f,
0.55971140f,0.59079570f,0.46185824f,0.59910893f,0.53943622f,0.38143632f,0.25197616f,0.32293645f,0.46860379f,0.05017685f,0.67394429f,0.40371668f,0.43617141f,0.62534189f,0.71934336f,0.12299241f,
0.88844264f,0.43488142f,0.06168697f,0.22225979f,0.65186620f,0.81692976f,0.40240505f,0.06970824f,0.75558996f,0.56049901f,0.93733442f,0.08468056f,0.14747922f,0.82777077f,0.78628325f,0.70719063f,
0.41856647f,0.24814153f,0.30629957f,0.95800269f,0.62957782f,0.55827576f,0.28093913f,0.09818164f,0.60845256f,0.95488340f,0.50189835f,0.04462401f,0.58022529f,0.22124168f,0.16761643f,0.46866792f,
0.65612310f,0.22930339f,0.69092774f,0.30798930f,0.04623315f,0.09333279f,0.37769756f,0.80182308f,0.65383184f,0.31503198f,0.88650364f,0.80131108f,0.14280275f,0.67278695f,0.50850165f,0.56136924f,
0.92092848f,0.81480128f,0.51937193f,0.55050629f,0.37307698f,0.80031109f,0.64868796f,0.98152959f,0.75519449f,0.15058628f,0.02615358f,0.33541977f,0.37182796f,0.19376999f,0.80408770f,0.02795105f,
0.42307338f,0.49501541f,0.33594036f,0.46930656f,0.58834821f,0.71363789f,0.27112964f,0.24218003f,0.02866986f,0.15763330f,0.04349109f,0.17147262f,0.83042026f,0.55199277f,0.73284185f,0.75134873f,
0.36679402f,0.25221378f,0.30185503f,0.73987097f,0.05252486f,0.95054299f,0.72140056f,0.80771935f,0.10112926f,0.74755412f,0.14313912f,0.47295722f,0.94132411f,0.94722682f,0.50090826f,0.36439753f,
0.44224223f,0.83684862f,0.83370405f,0.03059045f,0.55048651f,0.10483371f,0.27277049f,0.57915640f,0.26246703f,0.31626159f,0.75062901f,0.09288727f,0.86825436f,0.48347086f,0.84423602f,0.23504835f,
0.73568463f,0.14609104f,0.97491932f,0.78820950f,0.09663403f,0.69631988f,0.59592885f,0.19776329f,0.44387403f,0.73906797f,0.67072052f,0.38519818f,0.68629479f,0.17162879f,0.74959570f,0.12853700f,
0.00847740f,0.58329976f,0.15912746f,0.55896389f,0.68813348f,0.43189794f,0.13812031f,0.95060050f,0.74815953f,0.88874930f,0.04348776f,0.61641389f,0.37222016f,0.88772374f,0.85146224f,0.10790478f,
0.03381480f,0.82638156f,0.89611429f,0.13044883f,0.52270150f,0.49204311f,0.32821214f,0.96657550f,0.23111106f,0.99893266f,0.35177371f,0.91740584f,0.17056143f,0.10136939f,0.04594284f,0.17903884f,
0.68466914f,0.20507030f,0.73800278f,0.37280262f,0.63696826f,0.87612307f,0.32340312f,0.38512778f,0.76487237f,0.36689088f,0.00154165f,0.13709252f,0.25461462f,0.85300386f,0.24499729f,0.28842944f,
};


__device__ __host__ const float3 Roberts2(const int n) {
    const float g = 1.32471795724474602596;
    const float3 a = float3{ 1.0f / g, 1.0f / (g * g) };
    return  frac(a * n + 0.5);
}

__device__ __host__ const float3 UniformSampleHemisphere(const float x, const float y) {
    float Phi = float(2 * 3.14159265359 * x);
    float CosTheta = sqrt(float(y));
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

    float PDF = 1 / (4 * 3.14159265359f);

    return H;
}

__device__ __host__ float HenyeyGreenstein(float cos, float g)
{
    float g2 = g * g;
    return ((1.0f - g2) / pow((1.0f + g2 - 2.0f * g * cos), 1.5f)) * 0.25f /*/ 3.14159265359*/;
}

__device__ __host__ float HenyeyGreenstein_Avg(float3 ViewDir, float3 Dir,float Offset, float Radius, float g)
{
    float hg = 0.0;
    Radius = 0.5 * Radius;
    for (int i = 0; i < 128; i++)
    {
        float u = RandomSequence[i * 3 + 0];
        float v = RandomSequence[i * 3 + 1];
        float w = RandomSequence[i * 3 + 2];
        float theta = u * 2.0 * 3.1415926535;
        float phi = acos(2.0 * v - 1.0);
        float r = cbrt(w);
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);
        float sinPhi = sin(phi);
        float cosPhi = cos(phi);
        float x = Radius * r * sinPhi * cosTheta;
        float y = Radius * r * sinPhi * sinTheta;
        float z = Radius * r * cosPhi;
        float3 RandomPoint = float3{x,y,z} + Dir * Offset;
        float cos = dot(normalize(RandomPoint), ViewDir);
        hg += HenyeyGreenstein(cos, g);
    }
    return hg / 128.0;
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

__device__ __host__ float3 SampleHenyeyGreenstein(const int index, const float3 XMain, const float3 YMain, const float3 ZMain, const float g)
{
    float3 SpherePoint = SphereX[index % 64];
    float e0 = max(min(SpherePoint.z * 0.5f + 0.5f, 1.0), 0.0);
    float e1 = max(min((atan2(SpherePoint.y, SpherePoint.x) + 3.1415926535) / (2 * 3.14159265359), 1.0), 0.0);
    float CosTheta = SampleHeneyGreenstein(e0, g);
    float Phi = 2 * 3.14159265359 * e1;
    float SinTheta = sqrt(max(0.0f, 1.0f - CosTheta * CosTheta));

    float g2 = g * g;
    float hg = ((1.0f - g2) / pow((1.0f + g2 - 2.0f * g * CosTheta), 1.5f));// *0.25f / 3.14159265359;

    float e2 = RandomSequence[(index + 64) % 256];
    float alpha = (1.0 - exp(-1.0 / 256.0));//4.0 * 3.14159265359 * 
    float r = abs(-log(1.0 - hg * alpha * e2));
    return ((ZMain * (SinTheta * cos(Phi))) + (YMain * (SinTheta * sin(Phi))) + (XMain * CosTheta)) * r;
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

__device__ __host__ const float3 SphereRandom3(int index, float radius, float3 XMain, float3 YMain, float3 ZMain, float g)
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
__device__ __host__ const Offset_Layer_ GetSamples24_(int index)
{
    const float MsOffsetScale = 0.5;
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
}

__device__ __host__ const float3 HGRandom(int index, int& additionalMip, float3 XMain, float3 YMain, float3 ZMain, float g)
{
    int index3 = index * 3;
    float e0 = frac(RandomSequence[index3 + 0] + index * 1.539);
    float e1 = frac(RandomSequence[index3 + 1] + index * 2.277);
    float e2 = frac(RandomSequence[index3 + 2] + index * 3.782);

    float CosTheta = 0.0;
    float g2 = g * g;
    if (abs(g) < 0.0001)
    {
        CosTheta = e0 * 2.0 - 1.0;
    }
    else
    {
        float t0 = (1 - g2) / (1 - g + 2 * g * e0);
        CosTheta = (1 + g2 - t0 * t0) / (2 * g);
    }

    float Phi = 2 * 3.14159265359 * e1;
    float SinTheta = sqrt(max(0.0f, 1.0f - CosTheta * CosTheta));
    float3 SampleDir = (ZMain * (SinTheta * cos(Phi))) + (YMain * (SinTheta * sin(Phi))) + (XMain * CosTheta);

    const float DefualtRadius = 64.0/256.0;
    float HG = ((1.0f - g2) / pow((1.0f + g2 - 2.0f * g * CosTheta), 1.5f)) * 0.25f / 3.14159265359 /*/ 3.14159265359*/;
    float RMax = DefualtRadius * log(HG) / log(1.0 / (4.0 * 3.14159265359));
    float M = 1.0 / (1.0 - exp(-RMax));
    float SampleDist = max(-log(1.0 - (e2/M)),0.0);
    additionalMip = 2;// int(max(log2(256.0 * SampleDist + 1.5) - 1.0, 0.0));
    return SampleDir * SampleDist;
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