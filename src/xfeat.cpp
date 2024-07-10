#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>
#include "xfeat.h"

using namespace nvinfer1;

XFeat::XFeat(const std::string config):dev(torch::kCUDA)
{
    std::string engineFilePath = "/home/pranav/xfeat_ws/XFeatCPPTensorRT/weights/xfeat.engine";
    loadEngine(engineFilePath);
    context = std::unique_ptr<IExecutionContext, DestroyObjects> (engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("Failed to create execution context");
    }

    batchSize = 2;
    inputC = 1;
    inputH = 480;
    inputW = 640;
    outputH = inputH/8;
    outputW = inputW/8;
    _H = (outputH/32)*32;
    _W = (outputW/32)*32;

    rh = static_cast<float>(inputH) / static_cast<float>(_H);
    rw = static_cast<float>(inputW) / static_cast<float>(_W);

    inputIndex = engine->getBindingIndex("image");
    featsIndex = engine->getBindingIndex("feats");
    keypointsIndex = engine->getBindingIndex("keypoints");
    heatmapIndex = engine->getBindingIndex("heatmap");

}

void XFeat::detectAndCompute(const cv::Mat& img, cv::Mat& keypoints, cv::Mat& descriptors, cv::Mat& scores)
{
    int top_k = 4096;
    torch::Tensor input_Data = preprocessImages(img);
    featsData = torch::empty({batchSize,64,_H,_W}, torch::device(dev).dtype(torch::kFloat32));
    keypointsData = torch::empty({batchSize, 65, _H, _W}, torch::device(dev).dtype(torch::kFloat32));
    heatmapData = torch::empty({batchSize, 1, _H, _W}, torch::device(dev).dtype(torch::kFloat32));

    void* buffers[4]; 
    buffers[inputIndex] = input_Data.data_ptr();
    buffers[featsIndex] = featsData.data_ptr();
    buffers[keypointsIndex] = keypointsData.data_ptr();
    buffers[heatmapIndex] = heatmapData.data_ptr();

    
    context->executeV2(buffers);

    featsData = torch::nn::functional::normalize(featsData, torch::nn::functional::NormalizeFuncOptions().dim(1));
    keypointsData = get_kpts_heatmap(keypointsData);
    auto mkpts = NMS(keypointsData);

    auto _nearest = InterpolateSparse2d(mode = "nearest");
	auto bilinear = InterpolateSparse2d(mode = "bilinear");

    auto scores = (_nearest(keypointsData, mkpts, _H, _W) * _bilinear(heatmapData, mkpts, _H, _W)).squeeze(-1)

    featsData = featsData.cpu();
    keypointsData = keypointsData.cpu();
    heatmapData = heatmapData.cpu();    
    std::cout<<"All operations successful!"<<std::endl;
}

torch::Tensor XFeat::preprocessImages(const cv::Mat& img)
{
    torch::Tensor img_tensor = MatToTensor(img);

    img_tensor = torch::nn::functional::interpolate(
        img_tensor,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{_H, _W})
            .mode(torch::kBilinear)
            .align_corners(false)
    );


    return img_tensor;
}

std::vector<char> XFeat::readEngineFile(const std::string& engineFilePath)
{
    std::ifstream file(engineFilePath, std::ios::binary | std::ios::ate);
    if(!file.is_open()){
        throw std::runtime_error("Unable to open engine file: " + engineFilePath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0,std::ios::beg);

    std::vector<char> buffer(size);
    if(!file.read(buffer.data(),size)){
        throw std::runtime_error("Unable to read engine file: " + engineFilePath);
    }
    return buffer;
}

void XFeat::loadEngine(const std::string& engineFilePath)
{
    std::vector<char> engineData = readEngineFile(engineFilePath);

    runtime = std::unique_ptr<IRuntime,DestroyObjects>(createInferRuntime(gLogger));

    if(!runtime){
        throw std::runtime_error("Unable to create TensorRT runtime");
    }

    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    ICudaEngine* rawEngine = runtime->deserializeCudaEngine(engineData.data(),engineData.size());

    if(!rawEngine)
    {
        throw std::runtime_error("Unable to deserialize TensorRT engine");
    }
    engine = std::unique_ptr<ICudaEngine, DestroyObjects>(rawEngine);
}

inline torch::Tensor XFeat::MatToTensor(const cv::Mat& img)
{
    cv::Mat floatMat;
    img.convertTo(floatMat,CV_32F);

    CV_Assert(floatMat.isContinuous());

    int channels = floatMat.channels();
    int height = floatMat.rows;
    int width = floatMat.cols;

    torch::Tensor img_tensor = torch::from_blob(floatMat.data, {1, height, width, channels}, torch::kFloat32);
    img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();
    img_tensor = img_tensor.to(dev);

    return img_tensor;
}

torch::Tensor XFeat::NMS(const torch::Tensor& x, float threshold, int kernel_size)
{
    auto options = torch::TensorOptions().dtype(torch::kLong).device(dev);

    int B = x.size(0);
    int H = x.size(2);
    int W = x.size(3);
    int pad = kernel_size / 2;

    //Perform MaxPool2d
    auto local_max = torch::max_pool2d(x,kernel_size, 1, pad);

    // Compare x with local_max and threshold
    auto pos = (x == local_max) & (x > threshold);

    // Get the positions of the positive elements
    std::vector<torch::Tensor> pos_batched;
    pos_batched.reserve(B);
    for(int i = 0; i < B; i++)
    {
        pos_batched.emplace_back(torch::nonzero(pos[i]).flip(-1));
    }

    // Find the maximum number of keypoints to pad the tensor
    int pad_val = 0;
    for(const auto& tensor : pos_batched)
    {
        pad_val = std::max(pad_val, static_cast<int>(tensor.size(0)));
    }

    //Pad keypoints and build (B, N, 2) Tensor
    auto pos_tensor = torch::zeros({B, pad_val, 2}, options);
    for(int b = 0; b < B; b++)
    {
        pos_tensor[b].narrow(0, 0, pos_batched[b].size(0)) = pos_batched[b];
    }

    return pos_tensor;
}

torch::Tensor XFeat::get_kpts_heatmap(const torch::Tensor& kpts, float softmax_temp)
{
    //Apply softmax to the input tensor with temperature
    auto scores = torch::softmax(kpts * softmax_temp, 1).narrow(1,0,64);

    //Get dimension
    int B = scores.size(0);
    int H = scores.size(2);
    int W = scores.size(3);

    //Perform reshaping and permutation
    auto heatmap = scores.permute({0,2,3,1}).reshape({B, H, W, 8, 8});
    heatmap = heatmap.permute({0,1,3,2,4}).reshape({B, 1, H*8, W*8});

    return heatmap;
}
