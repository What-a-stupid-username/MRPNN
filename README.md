

# MRPNN - A neural volumetric renderer

![teaser](./pics/teaser.png)

## About
MRPNN is a neural volumetric renderer that can render high-order volumetric scattering in real time. See our paper for more details.
[SIG23: Deep Real-time Volumetric Rendering Using Multi-feature Fusion](https://sites.cs.ucsb.edu/~lingqi/publications/paper_mrpnn.pdf)

[Project page](http://www.cad.zju.edu.cn/home/jin/sig20231/SigMRPNN2023.htm) |
[Supplementary doc](http://www.cad.zju.edu.cn/home/jin/sig20231/Supplementary.pdf) |
[Dataset for demo](https://1drv.ms/f/s!AjOfZ7yWFdfGiElO457WE054P8Pt?e=pc3YDk)

![Renderer](./pics/cloud.gif)

## Install

### Requirements
- CUDA 11.x
	|Version|Compatible|Info|
	|-|-|-|
	|11.0|Yes*|CUDA11.0 Only support sm_75|
	|11.1|Yes|Recommended|
	|11.2|Yes|Recommended|
	|11.3|Yes*|**11.3 update 1** fix a compile error|
	|11.4~11.7|Not Sure||
	|11.8|No|Meet strange behavior|
- Glfw3 and GLEW if you want to compile with GUI
- x64 Ubuntu or Windows
- Nvidia GPU that supports sm_75 or sm_86

**If the render result is strange and incorrect, try to use CUDA 11.0 and disable the `RTX30XX` option no matter what card you are using. If you know why this happened, please help us fix it!**
Test with Geforce RTX2000 series, RTX3000 series, and Quadro A5000 GPU.

### To build on Linux:
- Install CUDA 11.x.
- Unzip 'tools/curand include.rar' into your cuda include directory.
- Run CMake to generate the project, and use `-T cuda=<PATH/TO/CUDA/toolkit>` to select a cuda version.
- If you are using the RTX3000 series (or another sm_86 card), enable the `RTX30XX` option.
- Enable the `RPNN` option if you want to compare with it.
- Compile your project, it may stuck for a while.
- Linux currently doesn't support build with GUI.

### To build on Windows:
- Install CUDA 11.x.
- Install glew and glfw3 if you need GUI support. (Remember to install the x64 version)
- Run CMake to generate the project, and use `-T cuda=<PATH/TO/CUDA/toolkit>` to select a cuda version.
- If you are using CUDA 11.0~11.3, you may also need to have MSVC v14.25 and using `-T version=14.25`.
- If you are using the RTX3000 series, enable the `RTX30XX` option.
- Enable the `RPNN` option if you want to compare with it.
- Check the `GUI` option if needed.
- Compile your project, it may stuck for a while.

## How to use

### Use test case
- Download test data from [OneDrive](https://1drv.ms/f/s!AjOfZ7yWFdfGiElO457WE054P8Pt?e=pc3YDk).
- Unzip them to `./TestCase/`.
- Run the **Test**.

*The bias comparison is only supported on Windows!*

### Use GUI
- Download [Cloud0.rar](https://1drv.ms/f/s!AjOfZ7yWFdfGiElO457WE054P8Pt?e=pc3YDk).
- Unzip it to `./TestCase/`.
- Build project with **GUI** option on.
- Run the **VolumeRender**.

### Use custom data 
```
VolumeRender v(/*volumeric data resolution*/512);

// filling volume with a sphere centered at (0.5,0.5,0.5)
v.SetDatas([](int x, int y, int z, float u, float v, float w) {
    float dis = distance(make_float3(0.5f, 0.5f, 0.5f), make_float3(u,v,w));
    return dis < 0.25 ? 1.0f : 0;
    });

// Call Update after changing volumetric data.
v.Update();

// Call render to generate result in memory (array of float3 colors)
v.Render(...)

// or create a camera of this volume
Camera cam(v);

// and use camera to render to file
cam.RenderToFile(...)
```

## BibTex
@inproceedings{ MRPNN:SIG:2023,

  author       = {Jinkai Hu and Chengzhong Yu and Hongli Liu and Ling-qi Yan and Yiqian Wu and Xiaogang Jin},
  
  title        = {Deep Real-time Volumetric Rendering Using Multi-feature Fusion},
  
  booktitle    = {{SIGGRAPH} '23: Special Interest Group on Computer Graphics and Interactive Techniques Conference, Los Angeles CA, United States, August 6 - 10, 2023},
  
  publisher    = {{ACM}},
  
  year         = {2023}
}
