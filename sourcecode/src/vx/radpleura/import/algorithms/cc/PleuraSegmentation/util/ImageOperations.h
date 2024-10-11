#ifndef IMAGEOPERATIONS_H
#define IMAGEOPERATIONS_H

#include <vector>
#include <iostream>


//Dlib includes
#include <dlib/array2d.h>
#include <dlib/image_transforms.h>

//ITK includes
#include <itkImage.h>
#include <itkRGBPixel.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkImageDuplicator.h>

namespace image
{

template <typename ImageT>
inline unsigned ExtractNeighborhoodITK(const typename ImageT::Pointer& inputImage,
                                       const typename ImageT::IndexType& centerIndex,
                                       const unsigned& neighborhoodSize,
                                       typename ImageT::Pointer& neighbohood)
{


    if(neighborhoodSize % 2 == 0)
    {
        std::cerr<<"ERROR: neighborhood must be odd"<<std::endl;
        return 0;
    }

    const unsigned& colum = centerIndex[0];
    const unsigned& row   = centerIndex[1];

    const auto imageSize = inputImage->GetRequestedRegion().GetSize();

    const auto imageRowSize = imageSize[1];
    const auto imageColSize = imageSize[0];


    unsigned  neighSizeCenter = neighborhoodSize/2;

    //Define begin index
    unsigned  imageRowBegin = (neighSizeCenter > row  ) ? 0 : row   - neighSizeCenter;
    unsigned  imageColBegin = (neighSizeCenter > colum) ? 0 : colum - neighSizeCenter;



    //Define actual neighbprhood size

    unsigned imageRowEnd = (row   + neighSizeCenter >= imageRowSize) ? imageRowSize :  row   + neighSizeCenter+1;
    unsigned imageColEnd = (colum + neighSizeCenter >= imageColSize) ? imageColSize :  colum + neighSizeCenter+1;



    //std::cout<<imageColEnd<<" -> "<<imageRowEnd<<std::endl;


    neighbohood =  ImageT::New();
    typename ImageT::RegionType region;
    region.SetSize({imageColEnd - imageColBegin, imageRowEnd - imageRowBegin});
    neighbohood->SetRegions(region);
    neighbohood->Allocate();


    //this is a slow version reimplement it using iterators
    unsigned roiR=0;
    for(unsigned r = imageRowBegin; r < imageRowEnd ; ++r)
    {
        unsigned roiC=0;
        for(unsigned c = imageColBegin; c < imageColEnd; ++c)
        {
            neighbohood->SetPixel({roiC++, roiR}, inputImage->GetPixel( {c, r} ));

        }
        ++roiR;
    }

    return 1;



}
/*
mask must be a binary image
0=Background
*/

template <typename ImageT, typename HistogramT>
inline void ComputeHistogram(const ImageT &image, const ImageT &mask, HistogramT & histogram)
{

    for(unsigned r = 0; r < image.nr() ; ++r)
    {
        for(unsigned c=0; c < image.nc(); ++c)
        {
            if(mask[r][c] != 0)
            {
                ++histogram[ static_cast<unsigned>(image[r][c]) ];
            }

        }

    }

}



/*
ImageT is a Dlib image type
index = col, row
*/
template <typename ImageT, typename IndexT = std::vector<unsigned long>>
inline unsigned ExtractNeighborhoodDLib(const ImageT& inputImage, const IndexT& centerIndex , const unsigned& neighborhoodSize, ImageT& roi)
{

    if(neighborhoodSize % 2 == 0)
    {
        std::cerr<<"ERROR: neighborhood must be odd"<<std::endl;
        return 0;
    }

    const unsigned& colum = centerIndex[0];
    const unsigned& row   = centerIndex[1];

    const auto imageRowSize = inputImage.nr();
    const auto imageColSize = inputImage.nc();

    unsigned  neighSizeCenter = neighborhoodSize/2;

    //Define begin index
    unsigned  imageRowBegin = (neighSizeCenter > row  ) ? 0 : row   - neighSizeCenter;
    unsigned  imageColBegin = (neighSizeCenter > colum) ? 0 : colum - neighSizeCenter;


    //Define actual neighbprhood size

    unsigned imageRowEnd = (row   + neighSizeCenter >= imageRowSize) ? imageRowSize :  row   + neighSizeCenter+1;
    unsigned imageColEnd = (colum + neighSizeCenter >= imageColSize) ? imageColSize :  colum + neighSizeCenter+1;


    /* std::cout<<index<<std::endl;
    std::cout<<imageColBegin<<" "<<imageRowBegin<<std::endl;
    std::cout<<imageColEnd - imageColBegin<<" "<<imageRowEnd - imageRowBegin<<std::endl;
    std::cout<<std::endl;
    */

    roi.clear();
    roi.set_size(imageRowEnd - imageRowBegin, imageColEnd - imageColBegin);
    unsigned roiR=0;
    for(unsigned r = imageRowBegin; r < imageRowEnd ; ++r)
    {
        unsigned roiC=0;
        for(unsigned c=imageColBegin; c < imageColEnd; ++c)
        {

            roi[roiR][roiC++] = inputImage[r][c];
        }
        ++roiR;
    }

    return 1;

}


template <typename DlibPixelT, typename ITKPixelT>
inline  void DlibToITK(const typename dlib::array2d<DlibPixelT>& dlibImage, typename itk::Image<ITKPixelT>::Pointer& itkImage)
{

    using itkImageT  = itk::Image<ITKPixelT,2>;


    unsigned cols = dlibImage.nc();
    unsigned rows  = dlibImage.nr();

    itkImage = itkImageT::New();
    typename itkImageT::RegionType region;
    region.SetSize({ cols, rows });
    itkImage->SetRegions(region);
    itkImage->Allocate();


    if constexpr(std::is_arithmetic<ITKPixelT>::value &&  std::is_arithmetic<DlibPixelT>::value) // is gray-level os scalar type
    {

        for(unsigned row = 0; row < dlibImage.nr(); ++row)
        {
            for(unsigned col = 0; col < dlibImage.nc(); ++col)
            {

                itkImage->SetPixel({col,row}, static_cast<ITKPixelT>(dlibImage[row][col]));

            }

        }
    }
    else
    {

        for(unsigned row = 0; row < dlibImage.nr(); ++row)
        {
            for(unsigned col = 0; col < dlibImage.nc(); ++col)
            {

                const auto & rgbPixel = dlibImage[row][col];

                const auto red   = static_cast<typename ITKPixelT::ComponentType>(rgbPixel.red  );
                const auto green = static_cast<typename ITKPixelT::ComponentType>(rgbPixel.green);
                const auto blue  = static_cast<typename ITKPixelT::ComponentType>(rgbPixel.blue );

                ITKPixelT pixel;
                pixel.Set(red, green, blue);
                itkImage->SetPixel({col,row}, pixel);

            }

        }

    }

    //io::printOK("Dlib image to ITK image");


}





template <typename ITKPixelT, typename DlibPixelT>
inline  void ITKToDlib(const typename itk::Image<ITKPixelT>::Pointer itkImage, typename dlib::array2d<DlibPixelT>& dlibImage)
{

    using itkImageT  = itk::Image<ITKPixelT,2>;
    using dlibImageT = dlib::array2d<DlibPixelT>;

    auto itkImageSize = itkImage->GetRequestedRegion().GetSize(); //cols, rows

    dlibImage.set_size(itkImageSize[1], itkImageSize[0]); //rows, cols

    itk::ImageRegionConstIteratorWithIndex<itkImageT> itkIt(itkImage, itkImage->GetRequestedRegion());


    if constexpr(std::is_arithmetic<ITKPixelT>::value &&  std::is_arithmetic<DlibPixelT>::value) // is gray-level os scalar type
    {
        for(;!itkIt.IsAtEnd(); ++itkIt)
        {
            const auto index = itkIt.GetIndex();
            dlibImage[index[1]][index[0]] = static_cast<DlibPixelT>(itkIt.Get());

        }
    }
    else //WARNING a cast to unsigned char is performed here
    {
        for(;!itkIt.IsAtEnd(); ++itkIt)
        {
            const auto index = itkIt.GetIndex();
            const auto pixel = itkIt.Get();
            dlibImage[index[1]][index[0]].red   = static_cast<unsigned char>(pixel[0]);
            dlibImage[index[1]][index[0]].green = static_cast<unsigned char>(pixel[1]);
            dlibImage[index[1]][index[0]].blue  = static_cast<unsigned char>(pixel[2]);

        }


    }


    //io::printOK("ITK image to DLib image");

}


}

#endif // IMAGEOPERATIONS_H
