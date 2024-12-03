#include "xfeat.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include "utils.h"
#include <filesystem>

int main(int argc, char* argv[])
{
    // Parse arguments
    if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <config.yaml> <engine_file>" << std::endl;
    return 1;
    }

    // Get the paths from the command line arguments
    std::filesystem::path config_path = argv[1];
    std::filesystem::path engine_path = argv[2];

    // Instantitate an XFeat object.
    XFeat xfeat(config_path.string(), engine_path.string());
    cv::Mat img1 = cv::imread("Image1.png");
    cv::Mat img2 = cv::imread("Image2.png");
    cv::Mat output;
    std::vector<cv::DMatch> matches, inliers;
    std::vector<cv::KeyPoint> k1,k2;

    torch::Tensor feats1, keypoints1, idx1;
    torch::Tensor feats2, keypoints2, idx2;

    // Warm-up runs
    std::cout<<"Performing warm-up runs"<<std::endl;
    for (int i = 0; i < 10; ++i) {
        xfeat.detectDense(img1, keypoints1, feats1);
        xfeat.detectDense(img2, keypoints2, feats2);
    }

    // Synchronize to ensure all operations are complete before starting the timer
    torch::cuda::synchronize();

    // Start timing
    std::cout<<"Beginning inference benchmark"<<std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    xfeat.detectDense(img1, keypoints1, feats1);
    xfeat.detectDense(img2, keypoints2, feats2);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<"Inference benchmark done, beginning feature matching and outlier rejection"<<std::endl;
    xfeat.match(feats1,feats2,idx1, idx2);
    TensorsToDMatch(idx1,idx2,matches);
    TensorToKeypoints(keypoints1, k1);
    TensorToKeypoints(keypoints2, k2);
    reject_outliers(k1,k2,matches,inliers);

    std::cout<<"Matching done, generating output image and saving it in the build directory"<<std::endl;


    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    VisualizeMatching(img1, k1, img2, k2, inliers, output, duration.count());

    cv::imwrite("DenseOutput.png",output);
    std::cout<<"All operations successful!"<<std::endl;

    return 0;
}
