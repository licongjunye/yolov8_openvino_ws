#pragma once
#ifdef OV_YOLOV8_EXPORTS
#define OV_YOLOV8_API _declspec(dllexport)
#else
#define OV_YOLOV8_API _declspec(dllimport)
#endif

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <openvino/openvino.hpp> //openvino header file
#include <opencv2/opencv.hpp>    //opencv header file
using namespace cv;
using namespace std;
using namespace dnn;


// 定义输出结构体
typedef struct {
	float prob;
	cv::Rect rect;
	int classid;
}Object;


//定义类
class  YoloModel
{
public:
	YoloModel();
	~YoloModel();
	//检测
	bool LoadDetectModel(const string& xmlName, string& device);
	bool YoloDetectInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj);



private:
	ov::InferRequest infer_request_Detect;
	ov::CompiledModel compiled_model_Detect;


	//增加函数
    // Keep the ratio before resize
	void letterbox(const Mat& source, Mat& result);
};
