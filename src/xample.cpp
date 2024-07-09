#include "xfeat.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    std::string config_path = "config_file_path";
    XFeat xfeat(config_path);
    cv::Mat input, feats, keypoints, heatmap;
    xfeat.detectAndCompute(input, feats, keypoints, heatmap);
    return 0;
}
