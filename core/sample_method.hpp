#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace std;
struct DataSample
{
	float Density[160];
	float Tr[160];
	float HG[160];
	float Radiance = 0.0;
	float g = 0.0;
	float scatter = 1.0;
};
struct Offset_Layer
{
	float Offset = 0.0;
	float Layer = 0.0;
	int type = 0;
	int localindex = 0;
	Offset_Layer(float noff, float nl, int type_ = 0, int localindex_ = 0)
	{
		Offset = noff; Layer = nl; type = type_; localindex = localindex_;
	}
};
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
struct SampleMethod
{
	/*128+32=160*/
	static void GetSamples23(std::vector<Offset_Layer>& samples)
	{
		const float MsOffsetScale = 0.6;// 0.65;
		samples.clear();
		//sp0
		for (int i = 0; i < 8; i++)
		{
			float currentMip = 0.0;
			float offset = 1.0 / 256.0;//1.0
			offset *= MsOffsetScale;
			samples.push_back(Offset_Layer(offset, currentMip, 5, i));
		}
		////////////////////////////////
		//sp1
		for (int i = 0; i < 8; i++)
		{
			float currentMip = 1.0;
			float offset = 2.5 / 256.0;//2.5
			offset *= MsOffsetScale;
			samples.push_back(Offset_Layer(offset, currentMip, 5, 8));
		}
		////////////////////////////////
		//sp2
		for (int i = 0; i < 16; i++)
		{
			float currentMip = 2.0;//radius 2
			float offset = 5.5 / 256.0;//5.5
			offset *= MsOffsetScale;
			samples.push_back(Offset_Layer(offset, currentMip, 5, 8));
		}
		////////////////////////////////
		//sp3
		for (int i = 0; i < 16; i++)
		{
			float currentMip = 3.0;//radius 4
			float offset = 11.5 / 256.0;//收缩
			offset *= MsOffsetScale;
			samples.push_back(Offset_Layer(offset, currentMip, 5, 8));
		}
		////////////////////////////////
		//sp4
		for (int i = 0; i < 16; i++)
		{
			float currentMip = 4.0;//radius 8
			float offset = 23.5 / 256.0;//收缩
			offset *= MsOffsetScale;
			samples.push_back(Offset_Layer(offset, currentMip, 5, 8));
		}
		////////////////////////////////
		//sp5
		for (int i = 0; i < 32; i++)
		{
			float currentMip = 5.0;//radius 16
			float offset = 47.5 / 256.0;//收缩
			offset *= MsOffsetScale;
			samples.push_back(Offset_Layer(offset, currentMip, 5, 8));
		}
		////////////////////////////////
		//sp6
		for (int i = 0; i < 32; i++)
		{
			float currentMip = 6.0;//radius 32
			float offset = 95.5 / 256.0;//收缩
			offset *= MsOffsetScale;
			samples.push_back(Offset_Layer(offset, currentMip, 5, 8));
		}
		//////////////////////////////////////////
		//direct
		for (int i = 0; i < 8; i++)
		{
			float currentMip = 0.0;
			float offset = i + 1;//1,2,3,4,5,6,7,8
			offset /= 256.0;
			samples.push_back(Offset_Layer(offset, currentMip, 1, 8));
		}//8
		for (int i = 0; i < 8; i++)
		{
			float currentMip = 1.0;
			float offset = 8.5 + 1 + 2.0 * i;//9.5,11.5,13.5,15.5 ,17.5,19.5,21.5,23.5
			offset /= 256.0;
			samples.push_back(Offset_Layer(offset, currentMip, 1, 8));
		}//16
		for (int i = 0; i < 8; i++)
		{
			float currentMip = 2.0;
			float offset = 26.5 + 4.0 * i;//26.5,30.5,34.5,38.5 , 42.5,46.5,50.5,54.5
			offset /= 256.0;
			samples.push_back(Offset_Layer(offset, currentMip, 1, 8));
		}//24
		for (int i = 0; i < 8; i++)
		{
			float currentMip = 3.0;
			float offset = 60.5 + 8.0 * i;//60.5,68.5,76.5,84.5, 92.5,100.5,108.5,116.5
			offset /= 256.0;
			samples.push_back(Offset_Layer(offset, currentMip, 1, 8));
		}//32
	}
};
