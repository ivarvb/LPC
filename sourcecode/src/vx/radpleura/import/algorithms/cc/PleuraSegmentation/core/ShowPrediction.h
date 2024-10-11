#ifndef SHOWPREDICTION_H
#define SHOWPREDICTION_H

//itk includes
#include <itkImage.h>
#include <itkRGBPixel.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "../util/InputOutput.h"
/* #include "../util/VTKViewer.h" */
#include "../util/ImageOperations.h"

class ShowPrediction
{

    using RGBImageT = itk::Image<itk::RGBPixel<unsigned char>, 2>;
    using IndexT = RGBImageT::IndexType;
    using RGBImageP = RGBImageT::Pointer;

    using GrayImageT = itk::Image<unsigned, 2>;
    using GrayImageP = GrayImageT::Pointer;

    const GrayImageT::PixelType Background{0};
    const GrayImageT::PixelType Foreground{255};

public:
    ShowPrediction();


    void SetPleuraMaskPath(const std::string& path);
    void SetOutputPath(const std::string& path);
    void SetImagesPath(const std::string& path);
    void SetKernelSize(unsigned size);

    void ReadCSV(const std::string& fileName,
                 unsigned imageNameIndex,
                 unsigned centerColIndex,
                 unsigned centerRowIndex,
                 unsigned labelIndex,
                 unsigned predictionIndex);



    void WritePredictions();

    std::vector<IndexT> Centers;
    std::vector<std::string> ImageNames;
    std::vector<int> Labels;
    std::vector<int> Predictions;

    std::string PleuraMasksPath{""};
    std::string ImagesPath{""};
    std::string OutputPath{""};
    unsigned KernelSize=201; //CSV???


private:

    void OverlayPleuraKernel(RGBImageP& image, const GrayImageP& kernel, const IndexT& center);


};

#endif // SHOWPREDICTION_H
