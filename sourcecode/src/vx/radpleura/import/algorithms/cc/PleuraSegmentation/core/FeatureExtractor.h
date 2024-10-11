#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H
//SLT includes
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <cinttypes>


//local includes
#include "../util/InputOutput.h"
/* #include "../util/VTKViewer.h" */
#include "../util/ImageOperations.h"
//#include "../util/FractalDimensionCalculator.h"
#include "../util/Cast.h"
#include "../util/Math.h"


//itk includes
#include <itkImage.h>
#include <itkRGBPixel.h>
#include <itkNeighborhoodIterator.h>
#include <itkScalarImageToCooccurrenceMatrixFilter.h>
#include <itkHistogramToTextureFeaturesFilter.h>
#include <itkImageDuplicator.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkLabelImageToShapeLabelMapFilter.h>
#include <itkRelabelComponentImageFilter.h>
//#include <itkStochasticFractalDimensionImageFilter.h>
#include <itkLabelMapToRGBImageFilter.h>
#include <itkShapeLabelObject.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkPoint.h>
#include <itkInvertIntensityImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkScaleTransform.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkFlatStructuringElement.h>
#include <itkImageMomentsCalculator.h>
#include <itkImageMaskSpatialObject.h>
#include <itkLabelToRGBImageFilter.h>


//Dlib includes
#include <dlib/array2d.h>
#include <dlib/image_transforms.h>
#include <dlib/gui_widgets.h>

class FeatureExtractor
{

private:

    using RGBImageT = itk::Image<itk::RGBPixel<unsigned char>, 2>;
    using RGBImageP = RGBImageT::Pointer;

    using GrayImageT = itk::Image<unsigned, 2>;
    using GrayImageP = GrayImageT::Pointer;

    const unsigned Background{0};
    const unsigned Foreground{255};

    using FeaturesT  = dlib::matrix<double>;
    using FeaturesVectorT = std::vector<FeaturesT>;
    using CentersT  = std::vector<GrayImageT::IndexType>;

    const double Pleura    =  1;
    const double NonPleura = -1;

public:


    void SetBoundariesPath(const std::string& boundariesPath);
    void SetImagesPath(const std::string& imagesPath);
    void SetLabelsPath(const std::string& labelsPath);
    void SetKernelSize(const std::size_t& kernelSize);
    void SetMasksPath(std::string const& masksPath);
    void SetPleuraMasksPath(const std::string& path);
    void SetErodeRadius(int radius);
    FeatureExtractor();

    void Process();

    void WriteFeaturesCSV(const std::string& fileName, bool writeHeader=false) const;

private:

    //for fast implementation, I assume that boundaries and input images have same names
    std::string BoundariesPath{"."};
    std::string ImagesPath{"."}; //Original Images
    std::string LabelsPath{"."};
    std::size_t KernelSize{51};
    std::string MasksPath{""};
    std::string PleuraMasksPath{""};

    int ErodeRadius{20};//scale factor to reduce image masks


    float MaskScaleFactor{1.5};


    CentersT   CentersVector;
    FeaturesVectorT  FeaturesVector;
    std::vector<double> LabelsVector;
    std::vector<std::string> ImagesNames;
    std::vector<std::uint16_t> CentersNumberPerImage;

public:
    GrayImageP ScaleMaskObjectsSize(GrayImageP mask);
    GrayImageP MatchMaskObjecst(GrayImageP maskOriginal, GrayImageP maskScaled);
    GrayImageP ErodeMask(GrayImageP maskOriginal);
    GrayImageP FindROIMask(GrayImageP maskOriginal, GrayImageP maskScaledCentered);


    void FindCenters(GrayImageP boundaries, CentersT& centers);
    void FindCenters2(GrayImageP boundaries, CentersT& centers);
    void FindLabels(const RGBImageP& labelsImage, const CentersT& centersVector, std::vector<double>& labelsVector);

    void ComputeLBPFeatures(GrayImageP grayImage, GrayImageP masks, const CentersT& centers, std::vector<std::vector<unsigned long>>& featuresVector);
    //void FractalDimensionBoxCounting(const GrayImageP &, const CentersT& centers, std::vector<double>& fractalDim);
    void ComputeCooccurrenceMatrixFeatures(GrayImageP grayImage, GrayImageP maskImage, const CentersT& centers, std::vector<std::vector<double>> &featuresVector);

    void CreateFeaturesVector(const std::vector< double>& fractal,
                              const std::vector<std::vector<unsigned long>>& lbp,
                              const std::vector<std::vector<double>>& cooccurrence,
                              const std::vector<std::vector<double>>& moments,
                              FeaturesVectorT& featuresVector);

    void ComputeStatisticalMoments(const GrayImageP & image,
                                   const GrayImageP & mask,
                                   const CentersT& centers,
                                   std::vector<std::vector<double>>& moments);


    static std::string  deleteSlash(std::string const& path);


    //void ComputeShapeFeatures(GrayImageP boundaries, const CentersT& centers, FeaturesVectorT& featuresVector);
    //void StochastichFractalFeatures(GrayImageP grayImage, GrayImageP mask, const CentersT& centers, FeaturesVectorT& featuresVector);

};

#endif // FEATUREEXTRACTOR_H
