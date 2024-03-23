#include "ov_yolov8.h"

namespace fs = std::filesystem;


// 全局变量
std::vector<cv::Scalar> colors = { cv::Scalar(255, 0, 0) , cv::Scalar(0, 0, 255) , cv::Scalar(128, 128, 128)};
const std::vector<std::string> class_names = { "B1", "B2", "B3", "B4", "B5", "B7",
                                               "R1", "R2", "R3", "R4", "R5", "R7",
                                               "N1", "N2", "N3", "N4", "N5", "N7"
                                               "B8", "R8"};

YoloModel::YoloModel()
{
    std::string isMakeDataset_str;
    cfg.readConfigFile("../params/params.cfg", "ismakedataset", isMakeDataset_str);
    isMakeDataset = (isMakeDataset_str == "true");
    cfg.readConfigFile("../params/params.cfg", "makedatasetpath", makedatasetpath);

}
YoloModel::~YoloModel()
{

}

// =====================检测========================//
bool YoloModel::LoadDetectModel(const string& xmlName, string& device)
{
    //待优化，如何把初始化部分进行提取出来
   // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    compiled_model_Detect = core.compile_model(xmlName, device);

    // -------- Step 3. Create an Inference Request --------
    infer_request_Detect = compiled_model_Detect.create_infer_request();
   
    return true;
}


std::vector<std::string> YoloModel::YoloDetectInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj)
{
    std::vector<std::string> detectedLabels;

    int64 start = cv::getTickCount();
    // -------- Step 4.Read a picture file and do the preprocess --------
    // Preprocess the image
    Mat letterbox_img;
    letterbox(src, letterbox_img);
    float scale = letterbox_img.size[0] / 640.0;
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(640, 640), Scalar(), true);

    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model_Detect.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request_Detect.set_input_tensor(input_tensor);
    // -------- Step 6. Start inference --------
    infer_request_Detect.infer();

    // -------- Step 7. Get the inference result --------
    auto output = infer_request_Detect.get_output_tensor(0);
    auto output_shape = output.get_shape();
    int rows = output_shape[2];        //8400
    int dimensions = output_shape[1];  //84: box[cx, cy, w, h]+80 classes scores



    // -------- Step 8. Postprocess the result --------
    float* data = output.data<float>();
    Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,84]
    float score_threshold = 0.25;
    float nms_threshold = 0.5;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;

    // Figure out the bbox, class_id and class_score
    for (int i = 0; i < output_buffer.rows; i++) {
        Mat classes_scores = output_buffer.row(i).colRange(4, output_buffer.cols - 4);
        Point class_id;
        double maxClassScore;
        minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > score_threshold) {
            class_scores.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);

            int left = int((cx - 0.5 * w) * scale);
            int top = int((cy - 0.5 * h) * scale);
            int width = int(w * scale);
            int height = int(h * scale);

            boxes.push_back(Rect(left, top, width, height));
        }
    }
    //NMS
    std::vector<int> indices;
    NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, indices);

    // -------- Visualize the detection results -----------
    dst = src.clone();
    for (size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        int class_id = class_ids[index];
        cv::Scalar color_;
        if (class_names[class_id][0] == 'B') {
            color_ = colors[0];
        } else if (class_names[class_id][0] == 'R') {
            color_ = colors[1];
        } else if (class_names[class_id][0] == 'N') {
            color_ = colors[2];
        }else{
            color_ = cv::Scalar(255, 255, 255);
        }
        rectangle(dst, boxes[index], color_, 2, 8);
        std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]).substr(0, 4);
        // Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        // Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height + 5);
        // cv::rectangle(dst, textBox, color_, FILLED);
        putText(dst, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 1.5, color_, 5);

        std::string labellist = class_names[class_id];
        detectedLabels.push_back(labellist);
    }
    float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(dst, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);


    if(indices.size() && isMakeDataset)
    {
        // Append code to save results at the end of the function
        std::string image_path = makedatasetpath + "/images/" + std::to_string(file_number) + ".jpg";
        std::string label_path = makedatasetpath + "/labels/" + std::to_string(file_number) + ".txt";

        // Ensure directories exist (create them if they don't)
        fs::create_directories(makedatasetpath + "/images");
        fs::create_directories(makedatasetpath + "/labels");

        // 保存检测结果图像
        cv::imwrite(image_path, src);

        // 保存检测框和标签到文本文件
        std::ofstream outfile(label_path);
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];
            
            // 归一化边界框坐标
            float x_center = (box.x + box.width / 2.0) / src.cols;
            float y_center = (box.y + box.height / 2.0) / src.rows;
            float width_norm = box.width / static_cast<float>(src.cols);
            float height_norm = box.height / static_cast<float>(src.rows);
            
            // 写入YOLO格式的标签信息
            outfile << class_ids[idx] << " " << x_center << " " << y_center << " " << width_norm << " " << height_norm << std::endl;
        }
        outfile.close();

        // 更新文件编号以供下次调用时使用
        file_number++;
    }



    return detectedLabels;

    // return true;
}


void YoloModel::letterbox(const cv::Mat& source, cv::Mat& result)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
}