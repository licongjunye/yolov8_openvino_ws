#include <iostream>
#include "ov_yolov8.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>


int main(int argc, char* argv[])
{
	YoloModel yolomodel;
    string xmlName_Detect = "/home/hlc/ultralytics/runs/detect/v2/train8/weights/best.xml";
	// string xmlName_Cls = "./yolov8/model/yolov8n-cls.xml";
	// string xmlName_Seg = "./yolov8/model/yolov8n-seg.xml";
	// string xmlName_Pose = "./yolov8/model/yolov8n-Pose.xml";
	string device = "GPU";
	bool initDetectflag = yolomodel.LoadDetectModel(xmlName_Detect, device);
	// bool initClsflag = yolomodel.LoadClsModel(xmlName_Cls, device);
	// bool initSegflag = yolomodel.LoadSegModel(xmlName_Seg, device);
	// bool initPoseflag = yolomodel.LoadPoseModel(xmlName_Pose, device);
	if (initDetectflag == true)
	{
		cout << "检测模型初始化成功" << endl;
	}
	// if (initClsflag == true)
	// {
	// 	cout << "分类模型初始化成功" << endl;
	// }
	// if (initSegflag == true)
	// {
	// 	cout << "分割模型初始化成功" << endl;
	// }
	// if (initPoseflag == true)
	// {
	// 	cout << "姿态模型初始化成功" << endl;
	// }
	// 读取图像
    // Mat img_Detect = cv::imread("/home/hlc/code/model/pic/frame_1140.jpg");
	// Mat img_Cls = img_Detect.clone();
	// Mat img_Seg = img_Detect.clone();
	// Mat img_Pose = img_Detect.clone();

	VideoCapture cap(0); // 0 for the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the camera." << std::endl;
        return -1;
    }

	Mat frame;

	while(true)
	{
		cap >> frame; // Capture a new frame from the camera
        if (frame.empty()) {
            std::cerr << "Error: Could not capture a frame." << std::endl;
            break;
        }

		// 检测推理
		Mat dst_detect;
		double cof_threshold_detect  = 0.25;
		double nms_area_threshold_detect = 0.5;
		vector<Object> vecObj = {};
		bool InferDetectflag = yolomodel.YoloDetectInfer(frame, cof_threshold_detect, nms_area_threshold_detect, dst_detect, vecObj);
	
		namedWindow("dst_pose", WINDOW_NORMAL);
		imshow("dst_pose", dst_detect);
		if (waitKey(1) >= 0) break;
	}



	// // 分类推理
	// Mat dst_cls;
	// double cof_threshold_Cls = 0.25;
	// double nms_area_threshold_Cls = 0.5;
	// vector<Object> vecObj_cls = {};
	// bool InferClsflag = yolomodel.YoloClsInfer(img_Cls, cof_threshold_Cls, nms_area_threshold_Cls, dst_cls, vecObj_cls);


	// // 分割推理
	// Mat dst_seg;
	// double cof_threshold_Seg = 0.25;
	// double nms_area_threshold_Seg = 0.5;
	// vector<Object> vecObj_seg = {};
	// bool InferSegflag = yolomodel.YoloSegInfer(img_Seg, cof_threshold_Seg, nms_area_threshold_Seg, dst_seg, vecObj_seg);

	// // 姿态推理
	// Mat dst_pose;
	// double cof_threshold_Pose = 0.25;
	// double nms_area_threshold_Pose = 0.5;
	// vector<Object> vecObj_Pose = {};
	// bool InferPoseflag = yolomodel.YoloPoseInfer(img_Pose, cof_threshold_Pose, nms_area_threshold_Pose, dst_pose, vecObj_Pose);

	
	// waitKey(0);
	destroyAllWindows();
    return 0;
}