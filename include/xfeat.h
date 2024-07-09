#ifndef XFEAT_H_
#define XFEAT_H_

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <type_traits>


using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if(severity <= Severity::kWARNING){
            std::cout<<msg<<std::endl;
        }
    }
};

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

class XFeat
{
    public:
    XFeat(const std::string config_file);

    /**
    * @brief Function to perform inferencing on the TensorRT engine. It preprocesses the data, performs inference, postprocesses the outputs and returns them.
    * @param img The input image to perform inference on.
    * @param keypoints Detected keypoints.
    * @param descriptors Descriptors of the keypoints.
    * @param scores Confidence scores of the keypoints.
   */
    void detectAndCompute(const cv::Mat& img, cv::Mat& keypoints, cv::Mat& descriptors, cv::Mat& scores);

    private:

    /**
    * @brief Function to preprocess the input images before feeding it into the engine. Returns the preprocessed image.
    * @param img The input image to the engine.
    * @param use_fp16 If true, the input image will be converted to CV_16F.
   */
    cv::Mat preprocessImages(const cv::Mat& img, bool use_fp16);

    /**
    * @brief Function to read the .engine file and load it into a buffer.
    * @param engineFilePath Path to the engine file.
   */
    std::vector<char> readEngineFile(const std::string& engineFilePath);

    /**
    * @brief Function to initialize a runtime, deserialize the engine and initialize the engine object.
    * @param engineFilePath Path to the engine file.
   */
    void loadEngine(const std::string& engineFilePath);

    /**
    * @brief Function to create buffers on the GPU for input and outputs. Returns a pointer to the newly allocated buffer
    * @param size Size to be allocated to the buffer.
   */
    std::unique_ptr<void, DestroyObjects> createDeviceBuffer(size_t size);


    //TensorRT Engine variables
    std::unique_ptr<IRuntime, DestroyObjects> runtime;
    std::unique_ptr<ICudaEngine, DestroyObjects> engine;
    std::unique_ptr<IExecutionContext, DestroyObjects> context;

    //TensorRT Logger
    Logger gLogger;

    //Input data variables params
    int batchSize, inputC, inputH, inputW;

    //Binding index for all the data
    int inputIndex, featsIndex, keypointsIndex, heatmapIndex;

    //Output data variables params
    int outputH, outputW,_H,_W;
    float rh,rw;

    //Preprocessed Image
    cv::Mat preprocessedImage;

    //Tensor to store output data
    torch::Tensor featsData, keypointsData, heatmapData;

    //Torch device (Must be CUDA)
    torch::Device dev;

};

#endif //XFEAT_H