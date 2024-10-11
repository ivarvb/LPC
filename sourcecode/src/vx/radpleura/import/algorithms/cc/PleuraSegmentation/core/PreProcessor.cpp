#include "PreProcessor.h"

PreProcessor::PreProcessor()
{

}


void PreProcessor::SetInputDatasetPath(const std::string& dataSetPath)
{
    InputDatasetPath = (*dataSetPath.rbegin() == '/') ? dataSetPath.substr(0, dataSetPath.length()-1) : dataSetPath;
}

PreProcessor::GrayImageP PreProcessor::HistogramEqualization(GrayImageP grayImage, bool show)
{
    using AdaptiveHistogramEqualizationImageFilterType = itk::AdaptiveHistogramEqualizationImageFilter<GrayImageT>;
    AdaptiveHistogramEqualizationImageFilterType::Pointer adaptiveHistogramEqualizationImageFilter = AdaptiveHistogramEqualizationImageFilterType::New();

    adaptiveHistogramEqualizationImageFilter->SetAlpha(Alpha);
    adaptiveHistogramEqualizationImageFilter->SetBeta(Beta);

    AdaptiveHistogramEqualizationImageFilterType::ImageSizeType radius;
    radius.Fill(Radius);
    adaptiveHistogramEqualizationImageFilter->SetRadius(radius);

    adaptiveHistogramEqualizationImageFilter->SetInput(grayImage);

    adaptiveHistogramEqualizationImageFilter->Update();


    //rescaling because equalization sometimes return pixels range within 0-254
    using RescaleType = itk::RescaleIntensityImageFilter<GrayImageT, GrayImageT>;
    RescaleType::Pointer rescaler = RescaleType::New();
    rescaler->SetInput(adaptiveHistogramEqualizationImageFilter->GetOutput());
    rescaler->SetOutputMinimum(Background);
    rescaler->SetOutputMaximum(Foreground);
    rescaler->Update();


    auto outputImage = rescaler->GetOutput();

    if(show)
    {
        /* VTKViewer::visualize<GrayImageT>(outputImage, "Adaptive Histogram Equalization"); */
    }


    io::printOK("Adaptive Histogram Equalization");



    return outputImage;


}


void PreProcessor::SetLThreshold(float lThreshold)
{
    LThreshold = lThreshold;
}

void PreProcessor::SetBThreshold(float bThreshold)
{

    BThreshold = bThreshold;
}

void PreProcessor::SetAThreshold(float aThreshold)
{
    AThreshold = aThreshold;
}





/*
All background pixels are set to white

*/
PreProcessor::RGBImageP PreProcessor::ExtractForeground(const RGBImageP& inputImage, bool show)
{

    //float type image
    using FloatImageT = itk::Image<float,2>;
    using FloatImageP = FloatImageT::Pointer;

    auto outputImage = RGBImageT::New();
    outputImage->SetRegions(inputImage->GetRequestedRegion());
    outputImage->Allocate();


    //RGB to Lab
    using labImageT = itk::Image<itk::RGBPixel<float>, 2>;

    //RGB to XYZ
    using rgbToXyzFilterT = ColorConverterFilter<RGBImageT, labImageT>;
    auto rgbToXyzFilter = std::make_unique<rgbToXyzFilterT>();
    rgbToXyzFilter->setInput(inputImage);
    rgbToXyzFilter->rgbToXyz();

    //XYZ to LAB
    using xyzToLabFilterT = ColorConverterFilter<labImageT, labImageT>;
    auto xyzToLabFilter = std::make_unique<xyzToLabFilterT>();
    xyzToLabFilter->setInput(rgbToXyzFilter->getOutput());
    xyzToLabFilter->xyzToLab();

    auto labImage = xyzToLabFilter->getOutput();

    /*
    //computing max luminance value
    //Extracting L channel
    using ExtractChannelFilterT = ExtractChannelFilter<labImageT, FloatImageT>;
    std::unique_ptr<ExtractChannelFilterT> extractChannelFilter(new ExtractChannelFilterT());

    extractChannelFilter->setImputImage(xyzToLabFilter->getOutput());
    extractChannelFilter->extractChannel(0);

    auto lChannel = extractChannelFilter->getOutputImage();
    using ImageCalculatorFilterType = itk::MinimumMaximumImageCalculator<FloatImageT>;


    //Min max interpolation
    typename ImageCalculatorFilterType::Pointer imageCalculatorFilter = ImageCalculatorFilterType::New();
    imageCalculatorFilter->SetImage(lChannel);
    imageCalculatorFilter->Compute();
    float maxLuminance = imageCalculatorFilter->GetMaximum();
    float minLuminance = imageCalculatorFilter->GetMinimum();



    const auto& minMax = [minLuminance, maxLuminance](float value)
    {
        return  ( 100.f * ( value - minLuminance) ) / (maxLuminance - minLuminance) ;
    };
*/

    itk::ImageRegionConstIterator<RGBImageT> inputIt(inputImage, inputImage->GetRequestedRegion());
    itk::ImageRegionIterator<RGBImageT> outputIt(outputImage, outputImage->GetRequestedRegion());
    itk::ImageRegionIterator<labImageT> labIt(labImage, labImage->GetRequestedRegion());

    typename RGBImageT::PixelType white;
    white.Fill(Foreground);

    for(; !inputIt.IsAtEnd(); ++inputIt, ++labIt, ++outputIt)
    {

        auto labPixel = labIt.Get();

        //if(minMax(labPixel[0]) > LThreshold  &&  labPixel[1] < AThreshold &&  labPixel[2] > BThreshold)

        if(labPixel[2] < BThreshold) //B
        {
            if(labPixel[1] > AThreshold )//A
            {
                if(labPixel[0] < LThreshold)
                {
                    outputIt.Set(inputIt.Get());
                }
                else
                {
                    outputIt.Set(white);
                }
            }

            else
            {
                outputIt.Set(white);
            }

        }
        else
        {
            outputIt.Set(white);
        }

    }

    if(show)
    {

        /* VTKViewer::visualize<RGBImageT>(outputImage, "Clean Background"); */
    }

    return outputImage;


}

std::vector<PreProcessor::GrayImageP> PreProcessor::GetOutputs()
{
    return OutputImages;
}

void PreProcessor::SetOutputDatasetPath(const std::string& outputPath)
{
    //delete '/' at the end of the path
    OutputDatasetPath = ( (*outputPath.rbegin()) == '/') ? outputPath.substr(0, outputPath.length()-1) : outputPath;
}

void PreProcessor::SetExtractForegroundArgs(float lThreshold, float aThreshold, float bThreshold)
{
    LThreshold = lThreshold;
    AThreshold = aThreshold;
    BThreshold = bThreshold;
}
void PreProcessor::SetHistogramEqualizationArgs(float alpha, float beta, unsigned radius)
{
    Alpha  = alpha;
    Beta   = beta;
    Radius = radius;
}

void PreProcessor::Process(bool returnResults)
{

    if(InputDatasetPath==OutputDatasetPath)
    {
        std::cerr<<"Input Dataset path and Output Dataset Path MUST be different"<<std::endl;
        return;
    }

    //usings
    using rgbToGrayFilterType = itk::RGBToLuminanceImageFilter<RGBImageT, GrayImageT>;
    using RescaleType = itk::RescaleIntensityImageFilter<GrayImageT, GrayImageT>;

    //get all files in dataset path
    std::vector<std::string> imagePaths;
    std::vector<std::string> imageNames;
    // std::string imageName;
    for(auto filePath: std::filesystem::directory_iterator(InputDatasetPath))
    {
        imagePaths.push_back(filePath.path());
        imageNames.push_back(*io::Split( *io::Split(filePath.path(), '/').rbegin(), '.').begin() );
    }

    RGBImageP image;
    RGBImageP blurImage;
    RGBImageP cleanImage;
    GrayImageP equalizeImage;
    auto imagePathIt = imagePaths.begin();
    auto imageNameIt = imageNames.begin();
    for(; imagePathIt != imagePaths.end(); ++imagePathIt, ++imageNameIt)
    {
        io::printWait("Pre Processing image "+*imageNameIt);

        image = io::ReadImage<RGBImageT>(*imagePathIt);
        cleanImage = ExtractForeground(image, false);


 
        //rgb to gray
        auto rgbToGrayFilter = rgbToGrayFilterType::New();
        rgbToGrayFilter->SetInput(cleanImage);
        rgbToGrayFilter->Update();


        //To-Do rele
        equalizeImage = HistogramEqualization(rgbToGrayFilter->GetOutput(), false);

        if(returnResults)
        {
            OutputImages.push_back(equalizeImage);
        }



        io::WriteImage<GrayImageT>(equalizeImage, OutputDatasetPath+"/"+*imageNameIt+".tiff");
        //io::WriteImage<GrayImageT>(rgbToGrayFilter->GetOutput(), OutputDatasetPath+"/"+*imageNameIt+".tiff");
        //io::WriteImage<RGBImageT>(cleanImage, OutputDatasetPath+"/"+*imageNameIt+".tiff");

    }


}
