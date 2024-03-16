#include <iostream>
#include "ov_yolov8.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>


int main(int argc, char* argv[])
{
	YoloModel yolomodel;
    string xmlName_Detect = "/home/hlc/ultralytics/runs/detect/v2/train8/weights/best.xml";
	string device = "GPU";
	bool initDetectflag = yolomodel.LoadDetectModel(xmlName_Detect, device);

	if (initDetectflag == true)
	{
		cout << "检测模型初始化成功" << endl;
	}
	
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


	// waitKey(0);
	destroyAllWindows();
    return 0;
}