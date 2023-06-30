#pragma once


#ifdef LINUX
#undef GUI
#endif

#ifdef GUI

#include "camera.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <chrono>
#include <thread>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_functions.h>
#include <windows.h>

#include <Shlobj.h>
#pragma comment(lib,"shell32.lib")

using namespace std;

HWND GL_Window;
HWND DX_Window;
GLFWwindow* window = NULL;
int actual_frame = 0;

static GLFWwindow* init_opengl(int reso, string name)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(
        reso, reso, name.c_str(), NULL, NULL);
    if (!window) {
        fprintf(stderr, "Error creating OpenGL window.\n");;
        glfwTerminate();
    }
    glfwMakeContextCurrent(window);

    const GLenum res = glewInit();
    if (res != GLEW_OK) {
        fprintf(stderr, "GLEW error: %s.\n", glewGetErrorString(res));
        glfwTerminate();
    }

    glfwSwapInterval(0);
    GL_Window = GetActiveWindow();
    return window;
}

static void init_cuda()
{
    int cuda_devices[1];
    unsigned int num_cuda_devices;
    cudaGLGetDevices(&num_cuda_devices, cuda_devices, 1, cudaGLDeviceListAll);
    if (num_cuda_devices == 0) {
        fprintf(stderr, "Could not determine CUDA device for current OpenGL context\n.");
        exit(EXIT_FAILURE);
    }
    cudaSetDevice(cuda_devices[0]);
}
struct Window_context
{
    int zoom_delta;

    bool moving;
    double move_start_x, move_start_y;
    double move_dx, move_dy;
};

// GLFW scroll callback.
static void handle_scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    Window_context* ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
    if (yoffset > 0.0)
        ctx->zoom_delta = 1;
    else if (yoffset < 0.0)
        ctx->zoom_delta = -1;
}

// GLFW keyboard callback.
static void handle_key(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS) {
        Window_context* ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        default:
            break;
        }
    }
}
static void handle_mouse_button(GLFWwindow* window, int button, int action, int mods)
{
    Window_context* ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            ctx->moving = true;
            glfwGetCursorPos(window, &ctx->move_start_x, &ctx->move_start_y);
        }
        else
            ctx->moving = false;
    }
}
static void handle_mouse_pos(GLFWwindow* window, double xpos, double ypos)
{
    Window_context* ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
    if (ctx->moving)
    {
        ctx->move_dx += xpos - ctx->move_start_x;
        ctx->move_dy += ypos - ctx->move_start_y;
        ctx->move_start_x = xpos;
        ctx->move_start_y = ypos;
    }
}
static void handle_resize_window(GLFWwindow* window, int w, int h) {
    static int last_size = 0;
    if (w != h || w % 4 != 0) {
        int size = w != last_size ? w : h;
        size = size / 4 * 4;
        last_size = size;
        glfwSetWindowSize(window, size, size);
    }
}
static void resize_buffers(float3** accum_buffer_cuda, Histogram** histo_buffer_cuda, cudaGraphicsResource_t* display_buffer_cuda, GLuint tempFB, GLuint* tempTex, int width, int width2, GLuint display_buffer)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, display_buffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * width * 4, NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (*display_buffer_cuda)
        cudaGraphicsUnregisterResource(*display_buffer_cuda);
    cudaGraphicsGLRegisterBuffer(
        display_buffer_cuda, display_buffer, cudaGraphicsRegisterFlagsWriteDiscard);

    if (*accum_buffer_cuda)
        cudaFree(*accum_buffer_cuda);
    cudaMalloc(accum_buffer_cuda, width * width * sizeof(float3));

    if (*histo_buffer_cuda)
        cudaFree(*histo_buffer_cuda);
    cudaMalloc(histo_buffer_cuda, width * width * sizeof(Histogram));

    glDeleteTextures(1, tempTex);
    glBindFramebuffer(GL_FRAMEBUFFER, tempFB);
    glGenTextures(1, tempTex);
    glBindTexture(GL_TEXTURE_2D, *tempTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width2, width2, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, *tempTex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

}

static void update_camera(Camera& cam, double phi, double theta, float base_dist, int zoom)
{
    float3 cam_dir;
    cam_dir.x = float(-sin(phi) * sin(theta));
    cam_dir.y = float(-cos(theta));
    cam_dir.z = float(-cos(phi) * sin(theta));

    float dist = float(base_dist * pow(0.95, double(zoom)));

    float3 cam_pos = cam_dir * -dist;

    cam.SetPosition(cam_pos);
}

static GLint add_shader(GLenum shader_type, const char* source_code, GLuint program)
{
    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader, 1, &source_code, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    glAttachShader(program, shader);

    return shader;
}
static GLuint create_shader_program()
{
    GLint success;
    GLuint program = glCreateProgram();

    const char* vert =
        "#version 330\n"
        "in vec3 Position;"
        "out vec2 TexCoord;"
        "void main() {"
        "    gl_Position = vec4(Position, 1.0);"
        "    TexCoord = 0.5 * Position.xy + vec2(0.5);"
        "}";
    add_shader(GL_VERTEX_SHADER, vert, program);

    const char* frag =
        "#version 330\n"
        "in vec2 TexCoord;"
        "out vec4 FragColor;"
        "uniform sampler2D TexSampler;"
        "uniform int Size;"
        "uniform int FSR;"
        "uniform float sharp;"
        "void FsrEASU(out vec4 fragColor){    vec2 fragCoord = TexCoord * Size * 2;    vec4 scale = vec4(        1. / vec2(Size),        vec2(0.5f)    );    vec2 src_pos = scale.zw * fragCoord;    vec2 src_centre = floor(src_pos - .5) + .5;    vec4 f; f.zw = 1. - (f.xy = src_pos - src_centre);    vec4 l2_w0_o3 = ((1.5672 * f - 2.6445) * f + 0.0837) * f + 0.9976;    vec4 l2_w1_o3 = ((-0.7389 * f + 1.3652) * f - 0.6295) * f - 0.0004;    vec4 w1_2 = l2_w0_o3;    vec2 w12 = w1_2.xy + w1_2.zw;    vec4 wedge = l2_w1_o3.xyzw * w12.yxyx;    vec2 tc12 = scale.xy * (src_centre + w1_2.zw / w12);    vec2 tc0 = scale.xy * (src_centre - 1.);    vec2 tc3 = scale.xy * (src_centre + 2.);    vec3 col = vec3(        texture(TexSampler, vec2(tc12.x, tc0.y)).rgb * wedge.y +        texture(TexSampler, vec2(tc0.x, tc12.y)).rgb * wedge.x +        texture(TexSampler, tc12.xy).rgb * (w12.x * w12.y) +        texture(TexSampler, vec2(tc3.x, tc12.y)).rgb * wedge.z +        texture(TexSampler, vec2(tc12.x, tc3.y)).rgb * wedge.w    );    fragColor = vec4(col,1);}void FsrRCAS(float sharp, out vec4 fragColor){    vec2 uv = TexCoord;    vec3 col = texture(TexSampler, uv).xyz;    float max_g = col.y;    float min_g = col.y;    vec4 uvoff = vec4(1,0,1,-1)/Size;    vec3 colw;    vec3 col1 = texture(TexSampler, uv+uvoff.yw).xyz;    max_g = max(max_g, col1.y);    min_g = min(min_g, col1.y);    colw = col1;    col1 = texture(TexSampler, uv+uvoff.xy).xyz;    max_g = max(max_g, col1.y);    min_g = min(min_g, col1.y);    colw += col1;    col1 = texture(TexSampler, uv+uvoff.yz).xyz;    max_g = max(max_g, col1.y);    min_g = min(min_g, col1.y);    colw += col1;    col1 = texture(TexSampler, uv-uvoff.xy).xyz;    max_g = max(max_g, col1.y);    min_g = min(min_g, col1.y);    colw += col1;    float d_min_g = min_g;    float d_max_g = 1.-max_g;    float A;    max_g = max(0., max_g);    if (d_max_g < d_min_g) {        A = d_max_g / max_g;    } else {        A = d_min_g / max_g;    }    A = sqrt(max(0., A));    A *= mix(-.125, -.2, sharp);    vec3 col_out = (col + colw * A) / (1.+4.*A);    fragColor = vec4(col_out,1);}"
        "void main() {"
        "    if (FSR == 1) FsrEASU(FragColor);"
        "    else if (FSR == 2) FsrRCAS(sharp, FragColor);"
        "    else FragColor = texture(TexSampler, TexCoord);"
        "}";
    GLint fs = add_shader(GL_FRAGMENT_SHADER, frag, program);

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        fprintf(stderr, "Error linking shadering program\n");
        char info[10240];
        int len;
        glGetShaderInfoLog(fs, 10240, &len, info);
        fprintf(stderr, info);
        glfwTerminate();
    }

    glUseProgram(program);

    return program;
}

// Create a quad filling the whole screen.
static GLuint create_quad(GLuint program, GLuint* vertex_buffer)
{
    static const float3 vertices[6] = {
        { -1.f, -1.f, 0.0f },
        {  1.f, -1.f, 0.0f },
        { -1.f,  1.f, 0.0f },
        {  1.f, -1.f, 0.0f },
        {  1.f,  1.f, 0.0f },
        { -1.f,  1.f, 0.0f }
    };

    glGenBuffers(1, vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, *vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    GLuint vertex_array;
    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);

    const GLint pos_index = glGetAttribLocation(program, "Position");
    glEnableVertexAttribArray(pos_index);
    glVertexAttribPointer(
        pos_index, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);

    return vertex_array;
}





#include "Imgui/imgui.h"
#include "Imgui/imgui_impl_dx9.h"
#include "Imgui/imgui_impl_win32.h"
#include <d3d9.h>
#include <tchar.h>


class GUIs {

    ImVec2 next_Window_pos;

    int selected_obj = -1;

    ImVec4 back_ground_color = ImVec4(1.f, 0.55f, 1.f, 1.f);

public:
    ImVec2 needed_size;
    int frame;
    float G;
    float alpha;
    int ms;

    bool render_surface = false;
    float IOR = 1.66;

    float tr = 1;

    float3 scatter_rate = { 1, 1, 1 };

    float env_exp = 1;

    float exposure = 1;

    float fps = 0;

    int toneType = 2;

    float lighta, lighty;
    float3 lightColor;

    const Camera& camera;

    bool predict = true;

    bool mrpnn = true;

    bool pause = false;

    bool denoise = true;

    bool fsr = false;

    bool checkboard = false;

    float sharpness = 0.5;

    bool change_hdri = false;

    char saveName[100] = "Noname";
    bool need_save = false;

    bool inspector = false;

    string hdri_path = "";
private:

    string SelectedHDR()
    {
        TCHAR szBuffer[MAX_PATH] = { 0 };
        OPENFILENAME ofn = { 0 };
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = FindWindow("ConsoleWindowClass", NULL);
        ofn.lpstrFilter = _T("HDR�ļ�(*.hdr *.HDR)\0*.hdr\0");
        ofn.lpstrInitialDir = _T("D:\\Program Files");
        ofn.lpstrFile = szBuffer;
        ofn.nMaxFile = sizeof(szBuffer) / sizeof(*szBuffer);
        ofn.nFilterIndex = 0;
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_EXPLORER;
        BOOL bSel = GetOpenFileName(&ofn);
        if (bSel) {
            return string(szBuffer);
        }
        else {
            return "";
        }
    }

    void LeftLabelText(string label, string txt) {
        ImGui::Text(label.c_str()); ImGui::SameLine();
        ImGui::Text(txt.c_str());
    }

    void DrawMainMenu() {
        ImGui::Begin("Settings", 0, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);
        ImGui::SetWindowPos(ImVec2(0, 0));
        ImGui::SetWindowSize(ImVec2(340, fsr ? 240 : 220));
        next_Window_pos = ImGui::GetWindowPos() + ImVec2(0, ImGui::GetWindowHeight());
        needed_size = ImGui::GetWindowSize();

        ImGui::Text("Frame No.%d", frame);
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        ImGui::Text("Resolution : %d", w);
        ImGui::Text("Render Time: %.3f ms(%.2f FPS)", 1000.f / fps, fps);
        float3 pos = camera.GetPosition();
        ImGui::Text("Camera Pos: (%.5f, %.5f, %.5f) ", pos.x, pos.y, pos.z);
        float3 l;
        float altiangle = lighty;
        l.y = sin(altiangle);
        float aziangle = lighta;
        l.x = cos(aziangle) * cos(altiangle);
        l.z = sin(aziangle) * cos(altiangle);
        ImGui::Text("Light Dir: (%.5f, %.5f, %.5f) ", l.x, l.y, l.z);
        ImGui::Separator();

        if (pause) {
            if (ImGui::Button("Continue Render")) {
                pause = !pause;
            }
        }
        else {
            if (ImGui::Button("Pause Render")) {
                pause = !pause;
            }
        }
        bool changed = false;

        ImGui::SameLine();
        ImGui::Checkbox("FSR", &fsr);
        ImGui::SameLine();
        changed |= ImGui::Checkbox("Checkboard", &checkboard);
        if (fsr) ImGui::SliderFloat("Sharpness", &sharpness, 0, 1);

        ImGui::Text("Name:");
        ImGui::SameLine();
        ImGui::InputText("", saveName, 100);
        ImGui::SameLine();
        if (ImGui::Button("Save")) {
            need_save = true;
        }

        changed |= ImGui::Checkbox("Predict", &predict);
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine();
        if (!predict)
        {
            ImGui::Checkbox("Denoise", &denoise); 
        }
        else
        {
#ifdef CRPNN
			changed |= ImGui::Checkbox("MRPNN", &mrpnn);
#else
            bool mustBeMRPNN = true;
			ImGui::Checkbox("MRPNN", &mustBeMRPNN);
#endif
        }

        ImGui::SliderFloat("Camera Exposure", &exposure, 0, 2);
        
        ImGui::Checkbox("Inspector", &inspector);
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine(); ImGui::Spacing();
        ImGui::SameLine();
        if (ImGui::Button("ToneMap"))
            ImGui::OpenPopup("my_ToneMap_popup");
        ImGui::SameLine();
        ImGui::TextUnformatted(toneType == 0 ? "None" : (toneType == 1 ? "Gamma" : "ACES"));
        if (ImGui::BeginPopup("my_ToneMap_popup"))
        {
            if (ImGui::Selectable("None"))
                toneType = 0;
            if (ImGui::Selectable("Gamma"))
                toneType = 1;
            if (ImGui::Selectable("ACES"))
                toneType = 2;
            ImGui::EndPopup();
        }

        if (changed)
            frame = 0;

        ImGui::End();
    }

    void DrawInspector() {
        DrawWindowLeftColum("Inspector", render_surface ? 532 : 510);

        bool changed = false;

        changed |= ImGui::Checkbox("Render Surface", &render_surface);
        if (render_surface) {
            changed |= ImGui::SliderFloat("IOR", &IOR, 1, 3);
        }
        changed |= ImGui::SliderFloat("G", &G, 0, 0.857);
        changed |= ImGui::SliderFloat("Alpha", &alpha, 0.1, 10);
        changed |= ImGui::SliderInt("Multi Scatter", &ms, 1, 1000);
        changed |= ImGui::SliderFloat("Tr scale", &tr, 1, 10);
        changed |= ImGui::SliderFloat3("Scatter rate", (float*)&scatter_rate, 0, 1);


        changed |= ImGui::SliderAngle("Light Azimuth", (float*)&lighta, -180, 180);
        changed |= ImGui::SliderAngle("Light Altitude ", (float*)&lighty, -90, 90);
        changed |= ImGui::ColorPicker3("Light Color", (float*)&lightColor, ImGuiColorEditFlags_::ImGuiColorEditFlags_Float | ImGuiColorEditFlags_::ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_::ImGuiColorEditFlags_HDR);

        changed |= ImGui::SliderFloat("Enviroment Exp", &env_exp, 0, 10);

        if (ImGui::Button("Select HDRI")) {
            hdri_path = SelectedHDR();
            if (hdri_path != "")
                change_hdri = true;
        }
        ImGui::SameLine();
        if (hdri_path != "")
            ImGui::Text(hdri_path.substr(hdri_path.find_last_of('\\') + 1).c_str());

        if (changed)
            frame = 0;

        ImGui::End();
    }

    bool right_click_on_item;

    void DrawWindowLeftColum(string name, int height, bool* show = NULL) {
        ImGui::Begin(name.c_str(), show, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);
        ImGui::SetWindowPos(next_Window_pos);
        ImGui::SetWindowSize(ImVec2(340, height));

        next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
        auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
        needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));
    }

public:
    GUIs(const Camera& cam): camera(cam) {  }

    void OnDrawGUI() {

        // Left column
        DrawMainMenu();
        if (inspector)
            DrawInspector();
    }
};


// Data
static LPDIRECT3D9              g_pD3D = NULL;
static LPDIRECT3DDEVICE9        g_pd3dDevice = NULL;
static D3DPRESENT_PARAMETERS    g_d3dpp = {};

// Forward declarations of helper functions
bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void ResetDevice();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Main code
int Main_Loop(GUIs& gui)
{
    // Create application window
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, _T("VRenderer Controller"), NULL };
    ::RegisterClassEx(&wc);
    RECT rect;
    GetWindowRect(GL_Window, &rect);
    HWND hwnd = ::CreateWindowEx(/*WS_EX_TOPMOST | */WS_EX_TRANSPARENT | WS_EX_LAYERED, wc.lpszClassName, _T("Controller"), WS_OVERLAPPEDWINDOW ^ WS_THICKFRAME, rect.right + 10, rect.top, 512, 1024, GL_Window, NULL, wc.hInstance, NULL);
    SetLayeredWindowAttributes(hwnd, 0, 1.0f, LWA_ALPHA);
    SetLayeredWindowAttributes(hwnd, 0, RGB(0, 0, 0), LWA_COLORKEY);
    LONG_PTR Style = ::GetWindowLongPtr(hwnd, GWL_STYLE);
    Style = Style & ~WS_CAPTION & ~WS_SYSMENU & ~WS_SIZEBOX;
    ::SetWindowLongPtr(hwnd, GWL_STYLE, Style);
    DWORD dwExStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
    dwExStyle &= ~(WS_VISIBLE);
    dwExStyle |= WS_EX_TOOLWINDOW;
    dwExStyle &= ~(WS_EX_APPWINDOW);
    SetWindowLong(hwnd, GWL_EXSTYLE, dwExStyle);
    //ShowWindow(hwnd, SW_SHOW);
    ShowWindow(hwnd, SW_HIDE);
    UpdateWindow(hwnd);

    DX_Window = hwnd;

    // Initialize Direct3D
    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClass(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer bindings
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX9_Init(g_pd3dDevice);

    SetWindowLong(hwnd, GWL_EXSTYLE, (GetWindowLong(hwnd, GWL_EXSTYLE) & ~WS_EX_TRANSPARENT) | WS_EX_LAYERED);

    auto& style = ImGui::GetStyle();
    style.FrameRounding = 12.f;
    style.GrabRounding = 12.f;

    bool first = true;
    float3 current_fpos;
    float3 velocity = { 0 };
    float rest_l;
    // Main loop
    MSG msg;
    ZeroMemory(&msg, sizeof(msg));
    while (msg.message != WM_QUIT)
    {
        if (GL_Window == 0) break;

        if (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplDX9_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        gui.OnDrawGUI();

        // Rendering
        ImGui::EndFrame();

        g_pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0, 1.0f, 0);
        if (g_pd3dDevice->BeginScene() >= 0)
        {
            ImGui::Render();
            ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
            g_pd3dDevice->EndScene();
        }

        HRESULT result = g_pd3dDevice->Present(NULL, NULL, NULL, NULL);

        if (first) {
            ShowWindow(hwnd, SW_SHOWDEFAULT);
            UpdateWindow(hwnd);
            RECT rect;
            GetWindowRect(DX_Window, &rect);
            current_fpos = float3{ (float)rect.left, (float)rect.top, 0 };
            GetWindowRect(GL_Window, &rect);
            float3 aim_ = float3{ (float)rect.right + 10, (float)rect.top, 0 };
            rest_l = distance(aim_, current_fpos);
            first = false;
        }

        //Handle loss of D3D9 device
        if (result == D3DERR_DEVICELOST && g_pd3dDevice->TestCooperativeLevel() == D3DERR_DEVICENOTRESET)
            ResetDevice();

        RECT rect, rect2;
        GetWindowRect(GL_Window, &rect);
        GetWindowRect(DX_Window, &rect2);

        float3 aim = float3{ (float)rect.right + 10, (float)rect.top, 0 };
        
        float l = distance(current_fpos, aim) - rest_l;
        float3 f;
        if (l == 0) f = float3{ 0,10,0 };
        else f = normalize(aim - current_fpos) * 10 * l + float3{ 0, 200, 0 };
        // static friction
        if (length(velocity) < 1) {
            f = normalize(f) * max(0.0f, length(f) - 1600); 
            velocity = float3 { 0 };
        }
        else {
            f = f + normalize(velocity) * -400;
        }

        float3 old_v = velocity;
        velocity = velocity + f / min(150.0f, max(0.01f, gui.fps));

        velocity = velocity * 0.99;

        current_fpos = aim;// current_fpos + (old_v + velocity) / 2 / min(150.0f, max(0.01f, gui.fps));

        MoveWindow(DX_Window, current_fpos.x, current_fpos.y, gui.needed_size.x, gui.needed_size.y, FALSE);

        static bool active = false;
        if (GetForegroundWindow() == GL_Window) {
            if (active == false) {
                active = true;
            }
            auto window = GetNextWindow(GetTopWindow(0), GW_HWNDNEXT);
        }
        else
        {
            active = false;
        }
    }

    ImGui_ImplDX9_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClass(wc.lpszClassName, wc.hInstance);

    return 0;
}

// Helper functions

bool CreateDeviceD3D(HWND hWnd)
{
    if ((g_pD3D = Direct3DCreate9(D3D_SDK_VERSION)) == NULL)
        return false;

    // Create the D3DDevice
    ZeroMemory(&g_d3dpp, sizeof(g_d3dpp));
    g_d3dpp.Windowed = TRUE;
    g_d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    g_d3dpp.BackBufferFormat = D3DFMT_UNKNOWN; // Need to use an explicit format with alpha if needing per-pixel alpha composition.
    g_d3dpp.EnableAutoDepthStencil = TRUE;
    g_d3dpp.AutoDepthStencilFormat = D3DFMT_D16;
    g_d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_ONE;           // Present with vsync
    //g_d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;   // Present without vsync, maximum unthrottled framerate
    if (g_pD3D->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWnd, D3DCREATE_HARDWARE_VERTEXPROCESSING, &g_d3dpp, &g_pd3dDevice) < 0)
        return false;

    return true;
}

void CleanupDeviceD3D()
{
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
    if (g_pD3D) { g_pD3D->Release(); g_pD3D = NULL; }
}

void ResetDevice()
{
    ImGui_ImplDX9_InvalidateDeviceObjects();
    HRESULT hr = g_pd3dDevice->Reset(&g_d3dpp);
    if (hr == D3DERR_INVALIDCALL)
        IM_ASSERT(0);
    ImGui_ImplDX9_CreateDeviceObjects();
}

// Win32 message handler
extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    if (msg == WM_SETFOCUS)
    {
        auto window = GetNextWindow(GetTopWindow(0), GW_HWNDNEXT);
        RECT rect;
        GetWindowRect(GL_Window, &rect);
        SetWindowPos(GL_Window, DX_Window, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, SWP_NOACTIVATE);
    }

    switch (msg)
    {
    case WM_SIZE:
        if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
        {
            g_d3dpp.BackBufferWidth = LOWORD(lParam);
            g_d3dpp.BackBufferHeight = HIWORD(lParam);
            ResetDevice();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}


void WriteBitmapFile(char* filename, int wid, int hei, unsigned char* bitmapData)
{
    int width = wid;
    int height = hei;

    BITMAPFILEHEADER bitmapFileHeader;
    memset(&bitmapFileHeader, 0, sizeof(BITMAPFILEHEADER));
    bitmapFileHeader.bfSize = sizeof(BITMAPFILEHEADER);
    bitmapFileHeader.bfType = 0x4d42;	//BM
    bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

    //���BITMAPINFOHEADER
    BITMAPINFOHEADER bitmapInfoHeader;
    memset(&bitmapInfoHeader, 0, sizeof(BITMAPINFOHEADER));
    bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmapInfoHeader.biWidth = width;
    bitmapInfoHeader.biHeight = height;
    bitmapInfoHeader.biPlanes = 1;
    bitmapInfoHeader.biBitCount = 24;
    bitmapInfoHeader.biCompression = BI_RGB;
    bitmapInfoHeader.biSizeImage = width * abs(height) * 3;

    FILE* filePtr;
    unsigned char tempRGB;
    int imageIdx;

    //swap R B
    for (imageIdx = 0; imageIdx < bitmapInfoHeader.biSizeImage; imageIdx += 3)
    {
        tempRGB = bitmapData[imageIdx];
        bitmapData[imageIdx] = bitmapData[imageIdx + 2];
        bitmapData[imageIdx + 2] = tempRGB;
    }

    char fname_bmp[128];
    sprintf(fname_bmp, "%s.bmp", filename);
    filePtr = fopen(fname_bmp, "wb");

    fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

    fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);

    fwrite(bitmapData, bitmapInfoHeader.biSizeImage, 1, filePtr);

    fclose(filePtr);
}
void SaveFB(GLFWwindow* window, char* fileName)
{
    GLubyte* pPixelData;
    GLint line_width;
    GLint PixelDataLength;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    line_width = width * 3; // �õ�ÿһ�е��������ݳ��� 
    line_width = (line_width + 3) / 4 * 4;

    PixelDataLength = line_width * height;

    // �����ڴ�ʹ��ļ� 
    pPixelData = (GLubyte*)malloc(PixelDataLength);
    if (pPixelData == 0)
        exit(0);


    // ��ȡ���� 
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    //glReadPixels(0, 0, img_w, img_h, GL_BGR_EXT, GL_UNSIGNED_BYTE, pPixelData);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pPixelData);

    WriteBitmapFile(fileName, line_width / 3, height, pPixelData);
    free(pPixelData);
}
#endif // GUI











void RunGUI(Camera& cam, VolumeRender& volume, float3 lightDir = normalize({ 1,1,1 }), float3 lightColor = { 1,1,1 }, float3 scatter_rate = { 1, 1, 1 }, float alpha = 1, float multiScatterNum = 10, float g = 0, bool hideConsole = false) {
#ifdef GUI
    if (hideConsole) {
        HWND hwnd;
        hwnd = FindWindow("ConsoleWindowClass", NULL);
        if (hwnd)
        {
            ShowOwnedPopups(hwnd, SW_HIDE);
            ShowWindow(hwnd, SW_HIDE);
        }
    }

    GUIs gui(cam);

    gui.G = g;
    gui.alpha = alpha;
    gui.ms = multiScatterNum;
    gui.lighta = atan2(lightDir.z, lightDir.x);
    gui.lighty = atan2(lightDir.y, sqrt(max(0.0001f, lightDir.x * lightDir.x + lightDir.z * lightDir.z)));
    gui.lightColor = lightColor;
    gui.scatter_rate = scatter_rate;

    volume.UpdateHGLut(g);

    Window_context window_context;
    memset(&window_context, 0, sizeof(Window_context));

    GLuint display_buffer = 0;
    GLuint display_tex = 0;
    GLuint program = 0;
    GLuint quad_vertex_buffer = 0;
    GLuint quad_vao = 0;
    int width = -1;
    int height = -1;

    // Init OpenGL window and callbacks.
    window = init_opengl(cam.resolution, cam.name);
    glfwSetWindowUserPointer(window, &window_context);
    glfwSetKeyCallback(window, handle_key);
    glfwSetScrollCallback(window, handle_scroll);
    glfwSetCursorPosCallback(window, handle_mouse_pos);
    glfwSetMouseButtonCallback(window, handle_mouse_button);
    glfwSetWindowSizeCallback(window, handle_resize_window);

    glGenBuffers(1, &display_buffer);
    glGenTextures(1, &display_tex);

    GLuint tempBuffer = 0;
    GLuint tempTex = 0;
    glGenFramebuffers(1, &tempBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, tempBuffer);
    glGenTextures(1, &tempTex);

    glBindTexture(GL_TEXTURE_2D, tempTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam.resolution, cam.resolution, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tempTex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    program = create_shader_program();
    quad_vao = create_quad(program, &quad_vertex_buffer);

    init_cuda();

    float3* accum_buffer = NULL;
    Histogram* histo_buffer_cuda = NULL;
    cudaGraphicsResource_t display_buffer_cuda = NULL;

    float3 cam_dir = cam.GetPosition();
    double theta = atan2(sqrt(max(0.0001f, cam_dir.x * cam_dir.x + cam_dir.z * cam_dir.z)), cam_dir.y);
    double phi = atan2(cam_dir.x, cam_dir.z);
    float base_dist = sqrt(dot(cam_dir, cam_dir));
    int zoom = 0;

    auto panel = thread([&]() {
        Main_Loop(gui);
        });

    while (!glfwWindowShouldClose(window)) {

        // Process events.
        glfwPollEvents();
        Window_context* ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
        if (ctx->move_dx != 0.0 || ctx->move_dy != 0.0 || ctx->zoom_delta) {

            zoom += ctx->zoom_delta;
            ctx->zoom_delta = 0;
            float M_PI = 3.1415926;
            phi -= ctx->move_dx * 0.001 * M_PI;
            theta -= ctx->move_dy * 0.001 * M_PI;
            theta = max(theta, 0.00 * M_PI);
            theta = min(theta, 1.00 * M_PI);
            ctx->move_dx = ctx->move_dy = 0.0;

            update_camera(cam, phi, theta, base_dist, zoom);
            gui.frame = 0;
        }

        // Reallocate buffers if window size changed.
        int nwidth, nheight;
        glfwGetFramebufferSize(window, &nwidth, &nheight); 
        if (gui.fsr) {
            nwidth /= 2; nheight /= 2;
        }
        if (nwidth != width || nheight != height)
        {
            width = nwidth;
            height = nheight;

            resize_buffers(
                &accum_buffer, &histo_buffer_cuda, &display_buffer_cuda, tempBuffer, &tempTex, width, gui.fsr ? width * 2 : width, display_buffer);
            //kernel_params.accum_buffer = accum_buffer;
            
            if (gui.fsr)
                glViewport(0, 0, width * 2, height * 2);
            else
                glViewport(0, 0, width, height);

            gui.frame = 0;

            //kernel_params.resolution.x = width;
            //kernel_params.resolution.y = height;
            //kernel_params.iteration = 0;

            // Allocate texture once
            glBindTexture(GL_TEXTURE_2D, display_tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }

        // Map GL buffer for access with CUDA.
        cudaGraphicsMapResources(1, &display_buffer_cuda, /*stream=*/0);
        void* p;
        size_t size_p;
        cudaGraphicsResourceGetMappedPointer(&p, &size_p, display_buffer_cuda);

        if (!gui.pause) {
            volume.SetEnvExp(gui.env_exp);
            volume.SetTrScale(gui.tr);
            volume.SetScatterRate(gui.scatter_rate * 1.001);
            volume.SetExposure(gui.exposure);
            volume.SetSurfaceIOR(gui.render_surface ? gui.IOR : -1);
            volume.SetCheckboard(gui.checkboard);

            if (gui.change_hdri) {
                volume.SetHDRI(gui.hdri_path);
                gui.change_hdri = false;
                gui.frame = 0;
            }

            float3 l;
            float altiangle = gui.lighty;
            l.y = sin(altiangle);
            float aziangle = gui.lighta;
            l.x = cos(aziangle) * cos(altiangle);
            l.z = sin(aziangle) * cos(altiangle);

            auto start_time = std::chrono::system_clock::now();

            cam.Render(accum_buffer, histo_buffer_cuda, reinterpret_cast<unsigned int*>(p), int2{ width , height }, gui.frame, l, gui.lightColor, gui.alpha, gui.ms, gui.G, gui.toneType, 
                gui.predict ? 
                    (gui.mrpnn ? VolumeRender::RenderType::MRPNN : VolumeRender::RenderType::RPNN)
                :
                    VolumeRender::RenderType::PT
                ,
                gui.denoise);

            auto finish_time = std::chrono::system_clock::now();
            float new_fps = 10000000.0f / (finish_time - start_time).count();
            gui.fps = lerp(gui.fps, new_fps, abs(new_fps - gui.fps) / gui.fps > 0.3 ? 1.0f : 0.01f);

            gui.frame++;
            actual_frame++;
        }

        // Unmap GL buffer.
        cudaGraphicsUnmapResources(1, &display_buffer_cuda, /*stream=*/0);

        // Update texture for display.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, display_buffer);
        glBindTexture(GL_TEXTURE_2D, display_tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Render the quad.
        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(quad_vao);

        glUniform1i(glGetUniformLocation(program, "Size"), width);
        if (gui.fsr) {
            glUniform1f(glGetUniformLocation(program, "sharp"), gui.sharpness);
            glBindFramebuffer(GL_FRAMEBUFFER, tempBuffer);
            glUniform1i(glGetUniformLocation(program, "FSR"), 1);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, tempTex);
            glUniform1i(glGetUniformLocation(program, "FSR"), 2);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }
        else {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glUniform1i(glGetUniformLocation(program, "FSR"), 0);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }

        glfwSwapBuffers(window);

        if (gui.need_save) {
            printf("saved.\n");
            SaveFB(window, gui.saveName);
            gui.need_save = false;
        }
    }

    cudaFree(accum_buffer);

    // Cleanup OpenGL.
    glDeleteVertexArrays(1, &quad_vao);
    glDeleteBuffers(1, &quad_vertex_buffer);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();

    GL_Window = 0;

    panel.join();

#else
    printf("GUI has not been Compiled.\n");
#endif // GUI
}