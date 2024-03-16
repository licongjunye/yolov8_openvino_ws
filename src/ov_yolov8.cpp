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


// =====================分类========================//
bool YoloModel::LoadClsModel(const string& xmlName, string& device)
{
    //待优化，如何把初始化部分进行提取出来
   // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    compiled_model_Detect_Cls = core.compile_model(xmlName, device);

    // -------- Step 3. Create an Inference Request --------
    infer_request_Cls = compiled_model_Detect_Cls.create_infer_request();

    return true;
}




bool YoloModel::YoloClsInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj)
{
    // -------- Step 4.Read a picture file and do the preprocess --------
    // Preprocess the image
    Mat letterbox_img;
    letterbox(src, letterbox_img);
    float scale = letterbox_img.size[0] / 640.0;
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(224, 224), Scalar(), true);

    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model_Detect_Cls.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request_Cls.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request_Cls.infer();

    // -------- Step 7. Get the inference result --------
    auto output = infer_request_Cls.get_output_tensor(0);
    auto output_shape = output.get_shape();

    // -------- Step 8. Postprocess the result --------
    float* output_buffer = output.data<float>();
    std::vector<float> result(output_buffer, output_buffer + output_shape[1]);
    auto max_idx = std::max_element(result.begin(), result.end());
    int class_id = max_idx - result.begin();
    float score = *max_idx;
    std::cout << "Class ID:" << class_id << " Score:" << score << std::endl;

    return true;
}


// =====================分割========================//
bool YoloModel::LoadSegModel(const string& xmlName, string& device)
{
    //待优化，如何把初始化部分进行提取出来
   // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    compiled_model_Seg = core.compile_model(xmlName, device);

    // -------- Step 3. Create an Inference Request --------
    infer_request_Seg = compiled_model_Seg.create_infer_request();

    return true;
}




bool YoloModel::YoloSegInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj)
{
    // -------- Step 4.Read a picture file and do the preprocess --------
    // Preprocess the image
    Mat letterbox_img;
    letterbox(src, letterbox_img);
    float scale = letterbox_img.size[0] / 640.0;
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(640, 640), Scalar(), true);

    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model_Seg.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request_Seg.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request_Seg.infer();

    // -------- Step 7. Get the inference result --------
    auto output0 = infer_request_Seg.get_output_tensor(0); //output0
    auto output1 = infer_request_Seg.get_output_tensor(1); //otuput1
    auto output0_shape = output0.get_shape();
    auto output1_shape = output1.get_shape();
    std::cout << "The shape of output0:" << output0_shape << std::endl;
    std::cout << "The shape of output1:" << output1_shape << std::endl;

    // -------- Step 8. Postprocess the result --------
    Mat output_buffer(output1_shape[1], output1_shape[2], CV_32F, output1.data<float>());    // output_buffer 0:x 1:y  2 : w 3 : h   4--84 : class score  85--116 : mask pos
    Mat proto(32, 25600, CV_32F, output0.data<float>()); //[32,25600] 1 32 160 160
    transpose(output_buffer, output_buffer); //[8400,116]
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;
    std::vector<Mat> mask_confs;
    // Figure out the bbox, class_id and class_score
    for (int i = 0; i < output_buffer.rows; i++) {
        Mat classes_scores = output_buffer.row(i).colRange(4, 84);
        Point class_id;
        double maxClassScore;
        minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > cof_threshold) {
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

            cv::Mat mask_conf = output_buffer.row(i).colRange(84, 116);
            mask_confs.push_back(mask_conf);
            boxes.push_back(Rect(left, top, width, height));
        }
    }
    //NMS
    std::vector<int> indices;
    NMSBoxes(boxes, class_scores, cof_threshold, nms_area_threshold, indices);

    // -------- Visualize the detection results -----------
    cv::Mat rgb_mask = cv::Mat::zeros(src.size(), src.type());
    cv::Mat masked_img;
    cv::RNG rng;

    Mat dst_temp = src.clone();
    for (size_t i = 0; i < indices.size(); i++) 
    {
        // Visualize the objects
        int index = indices[i];
        int class_id = class_ids[index];
        rectangle(dst_temp, boxes[index], colors_seg[class_id % 16], 2, 8);
        std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]).substr(0, 4);
        Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height + 5);
        cv::rectangle(dst_temp, textBox, colors_seg[class_id % 16], FILLED);
        putText(dst_temp, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));

        // Visualize the Masks
        Mat m = mask_confs[i] * proto;
        for (int col = 0; col < m.cols; col++) {
            sigmoid_function(m.at<float>(0, col), m.at<float>(0, col));
        }
        cv::Mat m1 = m.reshape(1, 160); // 1x25600 -> 160x160
        int x1 = std::max(0, boxes[index].x);
        int y1 = std::max(0, boxes[index].y);
        int x2 = std::max(0, boxes[index].br().x);
        int y2 = std::max(0, boxes[index].br().y);
        int mx1 = int(x1 / scale * 0.25);
        int my1 = int(y1 / scale * 0.25);
        int mx2 = int(x2 / scale * 0.25);
        int my2 = int(y2 / scale * 0.25);

        cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
        cv::Mat rm, det_mask;
        cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));

        for (int r = 0; r < rm.rows; r++) {
            for (int c = 0; c < rm.cols; c++) {
                float pv = rm.at<float>(r, c);
                if (pv > 0.5) {
                    rm.at<float>(r, c) = 1.0;
                }
                else {
                    rm.at<float>(r, c) = 0.0;
                }
            }
        }
        rm = rm * rng.uniform(0, 255);
        rm.convertTo(det_mask, CV_8UC1);
        if ((y1 + det_mask.rows) >= dst_temp.rows) {
            y2 = dst_temp.rows - 1;
        }
        if ((x1 + det_mask.cols) >= dst_temp.cols) {
            x2 = dst_temp.cols - 1;
        }

        cv::Mat mask = cv::Mat::zeros(cv::Size(dst_temp.cols, dst_temp.rows), CV_8UC1);
        det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
        add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);
        addWeighted(dst_temp, 0.5, rgb_mask, 0.5, 0, masked_img);
    }
    dst = masked_img.clone();

    return true;
}



// =====================姿态========================//
bool YoloModel::LoadPoseModel(const string& xmlName, string& device)
{
    //待优化，如何把初始化部分进行提取出来
   // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    compiled_model_Pose = core.compile_model(xmlName, device);

    // -------- Step 3. Create an Inference Request --------
    infer_request_Pose = compiled_model_Pose.create_infer_request();

    return true;
}




bool YoloModel::YoloPoseInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj)
{
    // -------- Step 4.Read a picture file and do the preprocess --------
    // Preprocess the image
    Mat letterbox_img;
    letterbox(src, letterbox_img);
    float scale = letterbox_img.size[0] / 640.0;
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(640, 640), Scalar(), true);

    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model_Pose.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request_Pose.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request_Pose.infer();

    // -------- Step 7. Get the inference result --------
    auto output = infer_request_Pose.get_output_tensor(0);
    auto output_shape = output.get_shape();

    // -------- Step 8. Postprocess the result --------
    float* data = output.data<float>();
    Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,56]
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;
    std::vector<std::vector<float>> objects_keypoints;

    // //56: box[cx, cy, w, h] + Score + [17,3] keypoints
    for (int i = 0; i < output_buffer.rows; i++) {
        float class_score = output_buffer.at<float>(i, 4);

        if (class_score > cof_threshold) {
            class_scores.push_back(class_score);
            class_ids.push_back(0); //{0:"person"}
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);
            // Get the box
            int left = int((cx - 0.5 * w) * scale);
            int top = int((cy - 0.5 * h) * scale);
            int width = int(w * scale);
            int height = int(h * scale);
            // Get the keypoints
            std::vector<float> keypoints;
            Mat kpts = output_buffer.row(i).colRange(5, 56);
            for (int i = 0; i < 17; i++) {
                float x = kpts.at<float>(0, i * 3 + 0) * scale;
                float y = kpts.at<float>(0, i * 3 + 1) * scale;
                float s = kpts.at<float>(0, i * 3 + 2);
                keypoints.push_back(x);
                keypoints.push_back(y);
                keypoints.push_back(s);
            }

            boxes.push_back(Rect(left, top, width, height));
            objects_keypoints.push_back(keypoints);
        }
    }
    //NMS
    std::vector<int> indices;
    NMSBoxes(boxes, class_scores, cof_threshold, nms_area_threshold, indices);

    dst = src.clone();
    // -------- Visualize the detection results -----------
    for (size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        // Draw bounding box
        rectangle(dst, boxes[index], Scalar(0, 0, 255), 2, 8);
        std::string label = "Person:" + std::to_string(class_scores[index]).substr(0, 4);
        Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height + 5);
        cv::rectangle(dst, textBox, Scalar(0, 0, 255), FILLED);
        putText(dst, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
        // Draw keypoints
        //std::vector<float> object_keypoints = objects_keypoints[index];
        //for (int i = 0; i < 17; i++)
        //{
        //    int x = std::clamp(int(object_keypoints[i * 3 + 0]), 0, dst.cols);
        //    int y = std::clamp(int(object_keypoints[i * 3 + 1]), 0, dst.rows);
        //    //Draw point
        //    circle(dst, Point(x, y), 5, posePalette[i], -1);
        //}
        // Draw keypoints-line

    }
    cv::Size shape = dst.size();
    plot_keypoints(dst, objects_keypoints, shape);
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



void YoloModel::sigmoid_function(float a, float& b) 
{
    b = 1. / (1. + exp(-a));
}



void YoloModel::plot_keypoints(cv::Mat& image, const std::vector<std::vector<float>>& keypoints, const cv::Size& shape)
{

    int radius = 5;
    bool drawLines = true;

    if (keypoints.empty()) {
        return;
    }

    std::vector<cv::Scalar> limbColorPalette;
    std::vector<cv::Scalar> kptColorPalette;

    for (int index : limbColorIndices) {
        limbColorPalette.push_back(posePalette[index]);
    }

    for (int index : kptColorIndices) {
        kptColorPalette.push_back(posePalette[index]);
    }

    for (const auto& keypoint : keypoints) {
        bool isPose = keypoint.size() == 51;  // numKeypoints == 17 && keypoints[0].size() == 3;
        drawLines &= isPose;

        // draw points
        for (int i = 0; i < 17; i++) {
            int idx = i * 3;
            int x_coord = static_cast<int>(keypoint[idx]);
            int y_coord = static_cast<int>(keypoint[idx + 1]);

            if (x_coord % shape.width != 0 && y_coord % shape.height != 0) {
                if (keypoint.size() == 3) {
                    float conf = keypoint[2];
                    if (conf < 0.5) {
                        continue;
                    }
                }
                cv::Scalar color_k = isPose ? kptColorPalette[i] : cv::Scalar(0, 0,
                    255);  // Default to red if not in pose mode
                cv::circle(image, cv::Point(x_coord, y_coord), radius, color_k, -1, cv::LINE_AA);
            }
        }
        // draw lines
        if (drawLines) {
            for (int i = 0; i < skeleton.size(); i++) {
                const std::vector<int>& sk = skeleton[i];
                int idx1 = sk[0] - 1;
                int idx2 = sk[1] - 1;

                int idx1_x_pos = idx1 * 3;
                int idx2_x_pos = idx2 * 3;

                int x1 = static_cast<int>(keypoint[idx1_x_pos]);
                int y1 = static_cast<int>(keypoint[idx1_x_pos + 1]);
                int x2 = static_cast<int>(keypoint[idx2_x_pos]);
                int y2 = static_cast<int>(keypoint[idx2_x_pos + 1]);

                float conf1 = keypoint[idx1_x_pos + 2];
                float conf2 = keypoint[idx2_x_pos + 2];

                // Check confidence thresholds
                if (conf1 < 0.5 || conf2 < 0.5) {
                    continue;
                }

                // Check if positions are within bounds
                if (x1 % shape.width == 0 || y1 % shape.height == 0 || x1 < 0 || y1 < 0 ||
                    x2 % shape.width == 0 || y2 % shape.height == 0 || x2 < 0 || y2 < 0) {
                    continue;
                }

                // Draw a line between keypoints
                cv::Scalar color_limb = limbColorPalette[i];
                cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), color_limb, 2, cv::LINE_AA);
            }
        }
    }
}



