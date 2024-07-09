#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>
#include <type_traits>


using namespace nvinfer1;


struct DestroyObjects {
     /**
    * @brief Custom struct to deal with freeing up the memory for Tensor objects, CUDA and CPU objects.
   */
    template <typename T>
    void operator()(T* ptr) const {
        if (ptr) {
            destroy(ptr);
        }
    }

private:
    // Enable if T has a destroy() member function (TensorRT objects). SFINAE (Substitution Failure Is Not An Error) 
    template <typename T>
    typename std::enable_if<std::is_member_function_pointer<decltype(&T::destroy)>::value>::type
    destroy(T* ptr) const {
        ptr->destroy();
    }

    // Fallback for CUDA memory (void*)
    void destroy(void* ptr) const {
        cudaFree(ptr);
    }
};

struct PreprocessedData {
    cv::Mat image;
    float rh;
    float rw;
};

PreprocessedData preprocessImages(const cv::Mat& img, bool use_fp16){
    cv::Mat floatImg;
    img.convertTo(floatImg, CV_32F);

    int H = floatImg.rows;
    int W = floatImg.cols;
    int _H = (H / 32) * 32;
    int _W = (W / 32) * 32;
    float rh = static_cast<float>(H) / _H;
    float rw = static_cast<float>(W) / _W;

    cv::resize(floatImg, floatImg, cv::Size(_W,_H), 0, 0, cv::INTER_LINEAR);

    if(use_fp16){
        floatImg.convertTo(floatImg, CV_16F);
    }
    return {floatImg, rh, rw};
}

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if(severity <= Severity::kWARNING){
            std::cout<<msg<<std::endl;
        }
    }
}gLogger;

std::vector<char> readEngineFIle(const std::string& engineFilePath)
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

ICudaEngine* loadEngine(const std::string& engineFilePath, IRuntime*& runtime){
    std::vector<char> engineData = readEngineFIle(engineFilePath);

    runtime = createInferRuntime(gLogger);

    if(!runtime){
        throw std::runtime_error("Unable to create TensorRT runtime");
    }

    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(),engineData.size());

    if(!engine)
    {
        throw std::runtime_error("Unable to deserialize CUDA engine");
    }
    return engine;
}

void* createDeviceBuffer(size_t size)
{
    void* buffer;
    cudaMalloc(&buffer,size);
    return buffer;
}

int main()
{
    std::string engineFilePath = "/home/pranav/xfeat_ws/XFeatCPPTensorRT/weights/xfeat.engine";
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = loadEngine(engineFilePath,runtime);

    IExecutionContext* context = engine->createExecutionContext();

    if(!context){
        throw std::runtime_error("Unable to create execution context");
    }

    int batchSize = 1;
    int inputC = 1;
    int inputH = 480;
    int inputW = 640;

    size_t inputSize = batchSize*inputC*inputH*inputW*sizeof(float);

    int outputH = inputH/8;
    int outputW = inputW/8;

    size_t featSize = batchSize*64*outputH*outputW*sizeof(float);
    size_t keypointSize = batchSize * 65 * outputH * outputW * sizeof(float);
    size_t heatmapSize = batchSize * 1 * outputH * outputW * sizeof(float);

    void* inputBuffer = createDeviceBuffer(inputSize);
    void* featsBuffer = createDeviceBuffer(featSize);
    void* keypointBuffer = createDeviceBuffer(keypointSize);
    void* heatmapBuffer = createDeviceBuffer(heatmapSize);

    void* buffers[] = {inputBuffer, featsBuffer, keypointBuffer, heatmapBuffer};

    //Create Data
    float* inputData = static_cast<float*>(malloc(inputSize));
     // Seed the random number generator
    std::srand(static_cast<unsigned>(time(0)));

    // Fill inputData with random values
    std::generate(inputData, inputData + (inputSize / sizeof(float)), []() {
        return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Random values between 0 and 1
    });

    cudaMemcpy(inputBuffer,inputData,inputSize,cudaMemcpyHostToDevice);

    float* featsData = static_cast<float*>(malloc(featSize));
    float* keypointsData = static_cast<float*>(malloc(keypointSize));
    float* heatmapData = static_cast<float*>(malloc(heatmapSize));

    
    context->executeV2(buffers);

    cudaMemcpy(featsData, featsBuffer,featSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(keypointsData, keypointBuffer,keypointSize,cudaMemcpyDeviceToHost);
    cudaMemcpy(heatmapData, heatmapBuffer,heatmapSize,cudaMemcpyDeviceToHost);

    cudaFree(inputBuffer);
    cudaFree(featsBuffer);
    cudaFree(keypointBuffer);
    cudaFree(heatmapBuffer);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    free(inputData);
    free(featsData);
    free(keypointsData);
    free(heatmapData);

    std::cout<<"All operations successful!"<<std::endl;
    return 0;
}
