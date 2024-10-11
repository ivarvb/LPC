#ifndef PIPE_H
#define PIPE_H
//STL includes
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <cinttypes>

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
#include "../util/ExtractChannelFilter.h"
#include "../core/PreProcessor.h"
#include "../core/BoundariesExtractor.h"
#include "../core/FeatureExtractor.h"

// declare types
using RGBPixelT = itk::RGBPixel<uint8_t>;
using RGBImageT = itk::Image<RGBPixelT,2>;
using RGBImageP = RGBImageT::Pointer;

using GrayImageT = itk::Image<unsigned, 2>;
using GrayImageP = GrayImageT::Pointer;


class Pipe{

private:

    //attributes
    //std::string inputImage{"."};
    std::string imagesPath{""};
    unsigned erodeRadius;

    /* 
    //background to white paramaters
    float lThreshold{98};
    float aThreshold{1};
    float bThreshold{-1};

    //Histogram equalization parameters
    float    alpha{1.f};
    float    beta{1.f};
    unsigned radius{5};
    */

public:

    /* Pipe(); */

    /* 
    void setInputImage(const std::string& dataSetPath);
    void setOutputPath(const std::string& outputPath); */
    void setPath(const std::string& outputPath);
    void setErodeRadius(unsigned arg);

    void execute();

};

// implements

void Pipe::setPath(const std::string& arg){
    imagesPath = (*arg.rbegin() == '/') ? arg.substr(0, arg.length()-1) : arg;
}
void Pipe::setErodeRadius(unsigned arg){
    erodeRadius = arg;
}


void Pipe::execute(){
    /* 
    * read image
    */
    RGBImageP image;
    GrayImageP image_pre;
    image = io::ReadImage<RGBImageT>(imagesPath+"/original.tiff");
    //std::filesystem::path p(inputImage);
    //std::string imageName= p.stem();




    // preprocessing
    //usings
    using rgbToGrayFilterType = itk::RGBToLuminanceImageFilter<RGBImageT, GrayImageT>;
    //using RescaleType = itk::RescaleIntensityImageFilter<GrayImageT, GrayImageT>;

    RGBImageP cleanImage;
    GrayImageP equalizeImage;

    PreProcessor preProcessor;
    preProcessor.SetLThreshold(98);
    preProcessor.SetBThreshold(-1);
    preProcessor.SetAThreshold(1);

    cleanImage = preProcessor.ExtractForeground(image, false);
 
    //rgb to gray
    auto rgbToGrayFilter = rgbToGrayFilterType::New();
    rgbToGrayFilter->SetInput(cleanImage);
    rgbToGrayFilter->Update();


    //To-Do rele
    equalizeImage = preProcessor.HistogramEqualization(rgbToGrayFilter->GetOutput(), false);
    io::WriteImage<GrayImageT>(equalizeImage, imagesPath+"/preprocessing.tiff");


    /**
    *boudary
    */
    BoundariesExtractor be;

    be.SetGaussSigma(0.5);
    be.SetSmallComponentsThreshold(500);
    be.SetThinBoundariesOff();

    GrayImageP boundaries;
    GrayImageP binaryFiltered;
    GrayImageP binaryImage;
    GrayImageP blur;
    GrayImageP mask;

    blur = be.GaussianBlur(equalizeImage, false);
    binaryImage = be.GrayToBinary(blur, false);
    binaryFiltered = be.DeleteSmallComponents(binaryImage);
    mask = be.ConnectForeground(binaryFiltered);

    boundaries = be.ExtractBoundaries(mask);
    
    io::WriteImage<GrayImageT>(boundaries, imagesPath+"/boundaries.tiff");
    io::WriteImage<GrayImageT>(mask, imagesPath+"/mask.tiff");




    /**
    *boudary mask - pleura mask
    */
    unsigned kernelSize = 50;
    unsigned erodeRadius = 30;
    FeatureExtractor fe;
    fe.SetKernelSize(kernelSize);
    fe.SetErodeRadius(erodeRadius);

    auto erodedMask = fe.ErodeMask(mask);
    auto pleuraMask = fe.FindROIMask(mask, erodedMask);

    io::WriteImage<GrayImageT>(pleuraMask, imagesPath+"/pleuramask.tiff");

}



#endif // PIPE_H
