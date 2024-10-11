#ifndef BOUNDARIESEXTRACTOR_H
#define BOUNDARIESEXTRACTOR_H

//ITK includes
#include <itkImage.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkBinaryContourImageFilter.h>
#include <itkBinaryThinningImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkLabelImageToShapeLabelMapFilter.h>
#include <itkRGBPixel.h>
#include <itkLabelMapToRGBImageFilter.h>
#include <itkRGBToLuminanceImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkSmoothingRecursiveGaussianImageFilter.h>


//SLT includes
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <cinttypes>


//local includes
#include "../util/InputOutput.h"
//#include "../util/VTKViewer.h"


class BoundariesExtractor
{

private:
    using GrayImageT = itk::Image<unsigned, 2>;
    using GrayImageP = GrayImageT::Pointer;

    const std::uint8_t Background{0};
    const std::uint8_t Foreground{255};

    //shape label map
    using LabelType = unsigned;
    using ShapeLabelObjectType = itk::ShapeLabelObject<LabelType, 2>;
    using LabelMapT = itk::LabelMap<ShapeLabelObjectType>;
    using LabelMapP = LabelMapT::Pointer;


public:
    BoundariesExtractor();

    void SetInputDatasetPath (const std::string& dataSetPath);
    void SetOutputDatasetPath(const std::string& outputPath);
    void SetOutputMaskPath(const std::string& outputPath);

    void SetThinBoundariesOn();
    void SetThinBoundariesOff();

    void SetGaussSigma(float sigma);

    void Process(bool returnResults=false);

    GrayImageP ExtractBoundaries(GrayImageP binaryImage);
    GrayImageP ThinningBoundaries(GrayImageP boundaries);


    void ConnectBackground(GrayImageP& grayImage);

    void SetSmallComponentsThreshold(std::uint16_t threshold);


private:
    //attributes
    std::string InputDatasetPath{"."};
    std::string OutputMaskPath{""};
    std::string OutputDatasetPath{""};

    bool ThinBoundaries{false};

    float  GaussSigma{1};

    std::uint16_t ComponentsThreshold{300};

    std::vector<GrayImageP> OutputImages;

public:
    GrayImageP GrayToBinary(GrayImageP grayImage, bool show=false);
    GrayImageP DeleteSmallComponents(GrayImageP edgesImage);

    GrayImageP LabelMapToBinaryImage(const LabelMapP& labelMap);
    GrayImageP GaussianBlur(const GrayImageP& inputImage, bool show=false);


    GrayImageP ConnectForeground(GrayImageP binaryImage);



};

#endif // BOUNDARIESEXTRACTOR_H
