#include "xfeat.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>

int main()
{
    std::string config_path = "config_file_path";
    XFeat xfeat(config_path);
    //cv::Mat img = cv::imread("path_to_image_file");
    int width = 640;
    int height = 480;

    // Create an empty Mat with the desired size and type
    cv::Mat randomImage(height, width, CV_32F);

    // Fill the Mat with random values between 0 and 1
    cv::randu(randomImage, 0, 1);
    torch::Tensor feats, keypoints, heatmap;
    xfeat.detectAndCompute(randomImage, feats, keypoints, heatmap);
    return 0;
}
