#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace std;
struct ReadDataList
{
	string Path;
	int BlurTimes = 0;
	int FeatureTimes = 0;
	int Type; //0 vox,1 sphere
	float AlphaMin = 1.0;
	float AlphaMax = 128.0;
	static void ReadConfig(vector<ReadDataList>& List, int& RDataSetSize, int& RLoopCount, int& RPointPerLoop, string& RDatasetName, const string& path)
	{
		std::ifstream Setting(path + "Setting.ini");
		if (!Setting) { std::cerr << "Failed to open. Terminating.\n"; exit(-1); }
		std::string line;
		std::string DataSetName;
		int DataSetSize = 0;
		int LoopCount = 0;
		while (!Setting.eof())
		{
			std::getline(Setting, line);
			ReadDataList NewList;
			if (line.substr(0, 1) == std::string("p"))//data set size
			{
				std::stringstream data(line);
				char c;
				data >> c >> DataSetSize;
			}
			else if (line.substr(0, 1) == std::string("m")) //model blurtime mixwithfbmtims
			{
				std::stringstream data(line);
				char c;
				data >> c >> NewList.Path >> NewList.BlurTimes >> NewList.FeatureTimes >> NewList.AlphaMin >> NewList.AlphaMax;
				List.push_back(NewList);
			}
			else if (line.substr(0, 1) == std::string("s"))//shape blurtime scaletime
			{
				std::stringstream data(line);
				char c;
				data >> c >> NewList.Path >> NewList.BlurTimes >> NewList.FeatureTimes >> NewList.AlphaMin >> NewList.AlphaMax;
				List.push_back(NewList);
			}
			else if (line.substr(0, 1) == std::string("d"))//shape blurtime scaletime
			{
				std::stringstream data(line);
				char c;
				data >> c >> DataSetName;
			}
		}
		Setting.close();
		printf("dataset size:%d\n", DataSetSize);
		for (auto& element : List)
		{
			printf("%s %d %d\n", element.Path.c_str(), element.BlurTimes, element.FeatureTimes);
			LoopCount += (element.BlurTimes + 1) * (element.FeatureTimes + 1);
		}
		int PointPerLoop = DataSetSize / LoopCount;
		printf("Point Per Loop:%d\n", PointPerLoop);
		RDataSetSize = DataSetSize;
		RLoopCount = LoopCount;
		RPointPerLoop = PointPerLoop;
		RDatasetName = DataSetName;
	}
};