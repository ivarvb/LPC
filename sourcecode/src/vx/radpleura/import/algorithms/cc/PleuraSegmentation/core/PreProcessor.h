#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H
//STL includes
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include<cinttypes>

//ITK includes
#include <itkImage.h>
#include <itkRGBPixel.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkRGBToLuminanceImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkAdaptiveHistogramEqualizationImageFilter.h>
#include <itkSmoothingRecursiveGaussianImageFilter.h>

//local includes
#include "../util/InputOutput.h"
#include "../util/ColorConverterFilter.h"
/* #include "../util/VTKViewer.h" */
#include "../util/ExtractChannelFilter.h"



class PreProcessor
{

private:
    //internal data types
    //suffix T=Type, P=Pointer

    using RGBPixelT = itk::RGBPixel<uint8_t>;
    using RGBImageT = itk::Image<RGBPixelT,2>;
    using RGBImageP = RGBImageT::Pointer;

    using GrayImageT = itk::Image<unsigned, 2>;
    using GrayImageP = GrayImageT::Pointer;

    const unsigned Background{0};
    const unsigned Foreground{255};


public:

    PreProcessor();

    void SetInputDatasetPath(const std::string& dataSetPath);
    void SetOutputDatasetPath(const std::string& outputPath);

    void SetExtractForegroundArgs(float lThreshold, float aThreshold, float bThreshold);
    void SetHistogramEqualizationArgs(float alpha, float beta, unsigned radius);

    void Process(bool returnResults=false);

    std::vector<GrayImageP> GetOutputs();

    void SetLThreshold(float lThreshold);
    void SetAThreshold(float aThreshold);
    void SetBThreshold(float bThreshold);



private:

    //attributes
    std::string InputDatasetPath{"."};
    std::string OutputDatasetPath{""};

    //background to white paramaters
    float LThreshold{98};
    float AThreshold{1};
    float BThreshold{-1};

    //Histogram equalization parameters
    float    Alpha{1.f};
    float    Beta{1.f};
    unsigned Radius{5};


    std::vector<GrayImageP> OutputImages;


public:
    //xx private methods
    RGBImageP  ExtractForeground(const RGBImageP& inputImage, bool show=false);
    GrayImageP HistogramEqualization(GrayImageP grayImage, bool show=false);





};

#endif // PREPROCESSOR_H
