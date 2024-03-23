#include <iostream>
#include "ov_yolov8.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include "../include//cfg.h"

int main(int argc, char* argv[])
{
    Cfg cfg;
    std::string xmlName_Detect;
    cfg.readConfigFile("../params/params.cfg", "xmlName_Detect", xmlName_Detect);
    std::string recordvideo_path;
    cfg.readConfigFile("../params/params.cfg", "recordvideo_path", recordvideo_path);
    std::string playvideo_path;
    cfg.readConfigFile("../params/params.cfg", "playvideo_path", playvideo_path);
    bool isplayvideo;
    std::string isplayvideo_str;
    cfg.readConfigFile("../params/params.cfg", "isplayvideo", isplayvideo_str);
    isplayvideo = (isplayvideo_str == "true");
    bool isrecordvideo;
    std::string isrecordvideo_str;
    cfg.readConfigFile("../params/params.cfg", "isrecordvideo", isrecordvideo_str);
    isrecordvideo = (isrecordvideo_str == "true");


    YoloModel yolomodel;
    std::string device = "GPU";
    bool initDetectflag = yolomodel.LoadDetectModel(xmlName_Detect, device);

    if (initDetectflag == true)
    {
        std::cout << "检测模型初始化成功" << std::endl;
    }

    cv::VideoCapture cap;
    if(isplayvideo)
    {
        cap.open(playvideo_path);
    }
    else
    {
        cap.open(0); // 0 for the default camera
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the camera." << std::endl;
        return -1;
    }

    cv::VideoWriter videoWriter;
    if(isrecordvideo)
    {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        // 设置视频输出对象
        videoWriter.open(recordvideo_path, fourcc, 30, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        if (!videoWriter.isOpened()) {
            std::cerr << "Error: Cannot create the output video file." << std::endl;
            return -1;
        }
    }


    cv::Mat frame;

    while(true)
    {
        cap >> frame; // Capture a new frame from the camera
        if (frame.empty()) {
            std::cerr << "Error: Could not capture a frame." << std::endl;
            break;
        }

        if(isrecordvideo)
            videoWriter.write(frame);

        // 检测推理
        cv::Mat dst_detect;
        double cof_threshold_detect  = 0.01;
        double nms_area_threshold_detect = 0.5;
        std::vector<Object> vecObj = {};
		std::vector<std::string> detectlabels = yolomodel.YoloDetectInfer(frame, cof_threshold_detect, nms_area_threshold_detect, dst_detect, vecObj);
		for (const auto& label : detectlabels) {
			std::cout <<"detectlabels:" << label << std::endl;
		}
        cv::namedWindow("dst_pose", cv::WINDOW_NORMAL);
        cv::resizeWindow("dst_pose", 1280, 960);
        cv::imshow("dst_pose", dst_detect);
        if (cv::waitKey(1) >= 0) break;

        // 检查用户是否按下了ESC键
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // 释放资源
    cap.release();

    if(isrecordvideo)
        videoWriter.release();

    cv::destroyAllWindows();
    return 0;
}
