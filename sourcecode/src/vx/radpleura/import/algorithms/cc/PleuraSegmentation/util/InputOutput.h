#ifndef INPUTOUTPUT_H
#define INPUTOUTPUT_H

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkImageFileWriter.h>

#include <string>
#include <sstream>
#include <vector>

#include "CustomPrint.h"

namespace io
{

static std::vector<std::string> Split(const std::string& inputStream, char delimiter)
{

    std::stringstream stringBuffer(inputStream);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(stringBuffer, token, delimiter))
    {
        tokens.push_back(std::move(token));

    }
    return tokens;

}


template<typename  ImageType>
static typename ImageType::Pointer ReadImage(const std::string& fileName)
{
    using readerT = itk::ImageFileReader<ImageType>;
    using readerP = typename readerT::Pointer;
    readerP reader = readerT::New();

    reader->SetFileName(fileName);
    reader->Update();
    return reader->GetOutput();
}

template<typename InputImageT>
inline void WriteImage(typename InputImageT::Pointer inputImage, const std::string& fileName, bool echo=false)
{


    using RGBImageT = itk::Image<itk::RGBPixel<unsigned char>, 2>;
    using InputPixelT = typename InputImageT::PixelType;

    if constexpr(std::is_arithmetic<InputPixelT>::value) // is gray-level
    {


        using RescaleType = itk::RescaleIntensityImageFilter<InputImageT, InputImageT>;
        typename RescaleType::Pointer rescale = RescaleType::New();
        rescale->SetInput(inputImage);
        rescale->SetOutputMinimum(itk::NumericTraits<InputPixelT>::Zero);
        rescale->SetOutputMaximum(itk::NumericTraits<InputPixelT>::Zero+255);
        rescale->Update();
        inputImage = rescale->GetOutput();

    }

    using FilterType = itk::CastImageFilter<InputImageT, RGBImageT>;
    typename FilterType::Pointer filter = FilterType::New();
    filter->SetInput(inputImage);

    using WriterType = itk::ImageFileWriter<RGBImageT>;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(fileName);
    writer->SetInput(filter->GetOutput());

    try
    {
        writer->Update();
        if(echo)
        {
            printOK("Writing Image");
        }
    }
    catch (itk::ExceptionObject & e)
    {
        std::cerr << "Error: Writing image "+fileName << e << std::endl;
        return;
    }


}

}

#endif // INPUTOUTPUT_H
