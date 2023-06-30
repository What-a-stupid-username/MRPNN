#include "omp.hpp"
#include <fstream>

void ParallelFill(float* datas, int resolution, FillFunc fillFunc) {
	#pragma omp parallel for
	for (int i = 0; i < resolution; i++)
		for (int j = 0; j < resolution; j++)
			for (int k = 0; k < resolution; k++)
			{
				float u, v, w;
				u = (i + 0.5) / resolution;
				v = (j + 0.5) / resolution;
				w = (k + 0.5) / resolution;
				float res = fillFunc(i, j, k, u, v, w);
				int index = (i * resolution + j) * resolution + k;
				datas[index] = res;
			}
}

void ParallelFor(float* result, int length, LoopFunc func) {
	#pragma omp parallel for
	for (int i = 0; i < length; i++) {
		result[i] = func(i);
	}	
}