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
#include "InterpolateSparse2D.h"


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
    void detectAndCompute(const cv::Mat& img, torch::Tensor& keypoints, torch::Tensor& descriptors, torch::Tensor& scores);

    /**
    * @brief Helper function to match the keypoints from two images based on descriptors cosine similarity.
    * @param feats1 Descriptors from image 1.
    * @param feats2 Descriptors from image 2.
    * @param matches Indices of matched points as a vector of cv::DMatches.
    * @param min_cossim Ratio of similarity between the cosines.
   */
    void match(const torch::Tensor& feats1, const torch::Tensor& feats2, std::vector<cv::DMatch>& matches, double min_cossim = 0.82);

    private:

    /**
    * @brief Function to preprocess the input images before feeding it into the engine. Returns the preprocessed image as a Tensor.
    * @param img The input image to the engine.
   */
    torch::Tensor preprocessImages(const cv::Mat& img);

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
    * @brief Helper function to convert an input image into a Tensor and store it on the GPU.
    * @param img Input image.
   */
    inline torch::Tensor MatToTensor(const cv::Mat& img);

    /**
    * @brief Helper function to perform non max suppression on a Tensor.
    * @param x Tensor containing the keypoints.
    * @param threshold Only consider keypoints above this threshold.
    * @param kernel_size Kernel size for the MaxPool2d operation.
   */
    torch::Tensor NMS(const torch::Tensor& x, float threshold = 0.05, int kernel_size = 5);

    /**
    * @brief Helper function to get the HeatMap from the keypoints Tensor.
    * @param kpts Tensor containing the keypoints.
    * @param softmax_temp Temperature to apply to the kpts in the SoftMax operation.
   */
    torch::Tensor get_kpts_heatmap(const torch::Tensor& kpts, float softmax_temp = 1.0);


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

    //Class objects to interpolate the Tensors
    InterpolateSparse2D _nearest;
	InterpolateSparse2D bilinear;

};

#endif //XFEAT_H