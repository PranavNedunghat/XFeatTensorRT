#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <torch/torch.h>

// Function to convert a keypoints tensor to std::vector<cv::KeyPoint>
static void TensorToKeypoints(const torch::Tensor& keypoints_tensor, std::vector<cv::KeyPoint>& keypoints_vector) 
{
    // Iterate through the tensor and create KeyPoint objects
    for (int i = 0; i < keypoints_tensor.size(0); ++i) {
        float x = keypoints_tensor[i][0].item<float>();
        float y = keypoints_tensor[i][1].item<float>();

        // Create a KeyPoint object
        cv::KeyPoint kp(x, y, 1.0f);

        // Add the KeyPoint object to the vector
        keypoints_vector.push_back(kp);
    }
}

//Convert Query Tensor and Train Tensor to cv::DMatch type
static void TensorsToDMatch(const torch::Tensor& idx1, const torch::Tensor& idx2, std::vector<cv::DMatch>& matches)
{
    for (int i = 0; i < idx1.size(0); ++i) 
        {
            int queryIdx = idx1[i].item<int>();
            int trainIdx = idx2[i].item<int>();
            float distance = 1.0;
            matches.emplace_back(queryIdx, trainIdx, distance);
        }
}

// Function to visualize the keypoints and matching, as well as display some benchmarking results.
static void VisualizeMatching(const cv::Mat &image0, const std::vector<cv::KeyPoint> &keypoints0, const cv::Mat &image1,
                              const std::vector<cv::KeyPoint> &keypoints1,
                              const std::vector<cv::DMatch> &_matches, cv::Mat &output_image, double cost_time = -1) {
    if(image0.size != image1.size) return;
    cv::drawMatches(image0, keypoints0, image1, keypoints1, _matches, output_image, cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));
    double sc = std::min(image0.rows / 640., 2.0);
    int ht = int(30 * sc);
    std::string title_str = "XFeat TensorRT";
    cv::putText(output_image, title_str, cv::Point(int(8*sc), ht), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, title_str, cv::Point(int(8*sc), ht), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    std::string feature_points_str = "Keypoints: " + std::to_string(keypoints0.size()) + ":" + std::to_string(keypoints1.size());
    cv::putText(output_image, feature_points_str, cv::Point(int(8*sc), ht*2), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, feature_points_str, cv::Point(int(8*sc), ht*2), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    std::string match_points_str = "Matches: " + std::to_string(_matches.size());
    cv::putText(output_image, match_points_str, cv::Point(int(8*sc), ht*3), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, match_points_str, cv::Point(int(8*sc), ht*3), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if(cost_time != -1) {
        std::string time_str = "FPS: " + std::to_string(1000 / cost_time);
        cv::putText(output_image, time_str, cv::Point(int(8 * sc), ht * 4), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc,
                    cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::putText(output_image, time_str, cv::Point(int(8 * sc), ht * 4), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

#endif //UTILS_H