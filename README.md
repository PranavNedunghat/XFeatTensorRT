# XFeatTensorRT
A C++ TensorRT implementation of XFeat: Accelerated Features deep learning model for lightweight image matching by VeRLab: https://github.com/verlab/accelerated_features

## Requirements
1) OpenCV 4
2) CUDA 12.2
3) NVIDIA TensorRT 8.6 +
4) LibTorch (PyTorch)
5) yaml-cpp

## How to build and run
1) Clone the repository 
  ```
  git clone https://github.com/PranavNedunghat/XFeatTensorRT.git
  ```
2) Modify the CMakeLists.txt file at lines: 23 and 28 which basically tell CMake where the LibTorch and TensorRT libraries are located. 
  ```
  find_package(Torch REQUIRED PATHS ${PROJECT_SOURCE_DIR}/libtorch)
  set(TENSORRT_ROOT ~/Downloads/TensorRT-8.6.1.6)
  ```
3) Create a build directory in the project directory and build the library and executables
  ```
  mkdir build && cd build
  cmake ..
  make
  ```
4) The build directory should contain the executables for both sparse and dense feature matching using XFeat. To run the executable:
  ```
  ./xample </path/to/config/file> </path/to/engine/file> <path/to/img1> <path/to/img2> #For the Sparse output
  ./xampleDense </path/to/config/file> </path/to/engine/file> <path/to/img1> <path/to/img2> #For the Dense output
  ```
The outputs will be located in the same build directory as SparseOutput.png and DenseOutput.png respectively

# Acknowledgements
This is a simple C++ implementation of XFeat: Accelerated Features deep learning model for lightweight image matching by VeRLab. If you find this useful please do support their incredible work:
1) **[GitHub](https://github.com/verlab/accelerated_features)**
2) **[Homepage](https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/)**
3) **[CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.html)**

Thanks also to **[IamShubhamGupto](https://github.com/verlab/accelerated_features/pull/4)** for his excellent work on exporting the model to onnx and the Python TensorRT implementation of XFeat. 
