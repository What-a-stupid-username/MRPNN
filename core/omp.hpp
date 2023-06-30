#pragma once

#include <functional>
#include <vector>
#include <string>
using namespace std;

typedef function<float(int x, int y, int z, float u, float v, float w)> FillFunc;

void ParallelFill(float* datas, int resolution, function<float(int x, int y, int z, float u, float v, float w)> fillFunc);

typedef function<float(int index)> LoopFunc;

void ParallelFor(float* result, int length, LoopFunc func);