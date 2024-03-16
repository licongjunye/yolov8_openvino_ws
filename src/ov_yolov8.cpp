#include "ov_yolov8.h"


// 全局变量
std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255) , cv::Scalar(0, 255, 0) , cv::Scalar(255, 0, 0) ,
                               cv::Scalar(255, 100, 50) , cv::Scalar(50, 100, 255) , cv::Scalar(255, 50, 100) };


std::vector<Scalar> colors_seg = { Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(170, 0, 255), Scalar(255, 0, 85),
                                   Scalar(255, 0, 170), Scalar(85, 255, 0), Scalar(255, 170, 0), Scalar(0, 255, 0),
                                   Scalar(255, 255, 0), Scalar(0, 255, 85), Scalar(170, 255, 0), Scalar(0, 85, 255),
                                   Scalar(0, 255, 170), Scalar(0, 0, 255), Scalar(0, 255, 255), Scalar(85, 0, 255) };

// 定义skeleton的连接关系以及color mappings
std::vector<std::vector<int>> skeleton = { {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7},
                                          {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7} };

std::vector<cv::Scalar> posePalette = {
        cv::Scalar(255, 128, 0), cv::Scalar(255, 153, 51), cv::Scalar(255, 178, 102), cv::Scalar(230, 230, 0), cv::Scalar(255, 153, 255),
        cv::Scalar(153, 204, 255), cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255), cv::Scalar(102, 178, 255), cv::Scalar(51, 153, 255),
        cv::Scalar(255, 153, 153), cv::Scalar(255, 102, 102), cv::Scalar(255, 51, 51), cv::Scalar(153, 255, 153), cv::Scalar(102, 255, 102),
        cv::Scalar(51, 255, 51), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 255)
};

std::vector<int> limbColorIndices = { 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16 };
std::vector<int> kptColorIndices = { 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9 };


const std::vector<std::string> class_names = { "B1", "B2", "B3", "B4", "B5", "B7",
                                               "R1", "R2", "R3", "R4", "R5", "R7"};



YoloModel::YoloModel()
{

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


bool YoloModel::YoloDetectInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj)
{
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
        rectangle(dst, boxes[index], colors[class_id % 6], 2, 8);
        std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]).substr(0, 4);
        Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height + 5);
        cv::rectangle(dst, textBox, colors[class_id % 6], FILLED);
        putText(dst, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
        std::cout<<label<<std::endl;
    }
    float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(dst, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
    return true;
}


void YoloModel::letterbox(const cv::Mat& source, cv::Mat& result)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
}