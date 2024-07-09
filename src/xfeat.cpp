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
    
    torch::Tensor input_Data = torch::randn({batchSize, inputC, _H, _W}, torch::device(dev).dtype(torch::kFloat32));
    featsData = torch::empty({batchSize,64,_H,_W}, torch::device(dev).dtype(torch::kFloat32));
    keypointsData = torch::empty({batchSize, 65, _H, _W}, torch::device(dev).dtype(torch::kFloat32));
    heatmapData = torch::empty({batchSize, 1, _H, _W}, torch::device(dev).dtype(torch::kFloat32));

    void* buffers[4]; 
    buffers[inputIndex] = input_Data.data_ptr();
    buffers[featsIndex] = featsData.data_ptr();
    buffers[keypointsIndex] = keypointsData.data_ptr();
    buffers[heatmapIndex] = heatmapData.data_ptr();

    
    context->executeV2(buffers);

    featsData = featsData.cpu();
    keypointsData = keypointsData.cpu();
    heatmapData = heatmapData.cpu();    
    std::cout<<"All operations successful!"<<std::endl;
}

cv::Mat XFeat::preprocessImages(const cv::Mat& img, bool use_fp16)
{
    return img;
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

inline std::unique_ptr<void, DestroyObjects> XFeat::createDeviceBuffer(size_t size)
{
    void* buffer;
    cudaError_t status = cudaMalloc(&buffer, size);
    if(status != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate device buffer");
    }
    return std::unique_ptr<void, DestroyObjects>(buffer);
}