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

	//分类
	bool YoloClsInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj);
	bool LoadClsModel(const string& xmlName, string& device);

	//分割
	bool LoadSegModel(const string& xmlName, string& device);
	bool YoloSegInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj);

	//姿态
	bool LoadPoseModel(const string& xmlName, string& device);
	bool YoloPoseInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj);


private:
	ov::InferRequest infer_request_Detect;
	ov::CompiledModel compiled_model_Detect;

	ov::InferRequest infer_request_Cls;
	ov::CompiledModel compiled_model_Detect_Cls;

	ov::InferRequest infer_request_Seg;
	ov::CompiledModel compiled_model_Seg;

	ov::InferRequest infer_request_Pose;
	ov::CompiledModel compiled_model_Pose;

	//增加函数
    // Keep the ratio before resize
	void letterbox(const Mat& source, Mat& result);
	void sigmoid_function(float a, float& b);
	void plot_keypoints(cv::Mat& image, const std::vector<std::vector<float>>& keypoints, const cv::Size& shape);
};
