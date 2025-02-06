#include "xfeat.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include "utils.h"
#include <filesystem>

int main(int argc, char* argv[])
{
    // Parse arguments
    if (argc != 5) {
    std::cerr << "Enter the path to the <config.yaml>, <engine_file>, <image_1>, <image_2>" << std::endl;
    return 1;
    }

    // Get the paths from the command line arguments
    std::filesystem::path config_path = argv[1];
    std::filesystem::path engine_path = argv[2];
    std::filesystem::path image1_path = argv[3];
    std::filesystem::path image2_path = argv[4];

    // Instantitate an XFeat object.
    XFeat xfeat(config_path.string(), engine_path.string());
    cv::Mat img1 = cv::imread(image1_path);
    cv::Mat img2 = cv::imread(image2_path);
    cv::Mat output;
    std::vector<cv::DMatch> matches, inliers;
    std::vector<cv::KeyPoint> k1,k2;

    torch::Tensor feats1, keypoints1, heatmap1, idx1;
    torch::Tensor feats2, keypoints2, heatmap2, idx2;

    // Warm-up runs
    std::cout<<"Performing warm-up runs"<<std::endl;
    for (int i = 0; i < 10; ++i) {
        xfeat.detectAndCompute(img1, keypoints1, feats1, heatmap1);
        xfeat.detectAndCompute(img2, keypoints2, feats2, heatmap2);
    }

    // Synchronize to ensure all operations are complete before starting the timer
    torch::cuda::synchronize();

    // Start timing
    std::cout<<"Beginning inference benchmark"<<std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    xfeat.detectAndCompute(img1, keypoints1, feats1, heatmap1);
    xfeat.detectAndCompute(img2, keypoints2, feats2, heatmap2);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<"Inference benchmark done, beginning feature matching and outlier rejection"<<std::endl;
    xfeat.match(feats1,feats2,idx1, idx2);
    TensorsToDMatch(idx1,idx2,matches);
    TensorToKeypoints(keypoints1, k1);
    TensorToKeypoints(keypoints2, k2);
    reject_outliers(k1,k2,matches,inliers);

    std::cout<<"Matching done, generating output image and saving it in the example_imgs directory"<<std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    VisualizeMatching(img1, k1, img2, k2, inliers, output, duration.count());

    std::filesystem::path output_path = image1_path.parent_path() / "SparseOutput.png";

    cv::imwrite(output_path,output);
    std::cout<<"All operations successful!"<<std::endl;

    return 0;
}
