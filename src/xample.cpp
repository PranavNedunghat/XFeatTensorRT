#include "xfeat.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include "utils.h"

int main()
{
    std::string config_path = "config_file_path";
    XFeat xfeat(config_path);
    cv::Mat img1 = cv::imread("Image1.png");
    cv::Mat img2 = cv::imread("Image2.png");
    cv::Mat output;
    std::vector<cv::DMatch> matches;

    torch::Tensor feats1, keypoints1, heatmap1;
    torch::Tensor feats2, keypoints2, heatmap2;

    // Warm-up runs
    for (int i = 0; i < 10; ++i) {
        xfeat.detectAndCompute(img1, keypoints1, feats1, heatmap1);
        xfeat.detectAndCompute(img2, keypoints2, feats2, heatmap2);
    }

    // Synchronize to ensure all operations are complete before starting the timer
    torch::cuda::synchronize();

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    xfeat.detectAndCompute(img1, keypoints1, feats1, heatmap1);
    xfeat.detectAndCompute(img2, keypoints2, feats2, heatmap2);

    // Synchronize again before stopping the timer
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    xfeat.match(feats1,feats2,matches);
    std::cout<<"All operations successful!"<<std::endl;

    std::vector<cv::KeyPoint> k1,k2;

    TensorToKeypoints(keypoints1, k1);
    TensorToKeypoints(keypoints2, k2);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    VisualizeMatching(img1, k1, img2, k2, matches, output, duration.count());

    cv::imwrite("Output.png",output);


    return 0;
}
