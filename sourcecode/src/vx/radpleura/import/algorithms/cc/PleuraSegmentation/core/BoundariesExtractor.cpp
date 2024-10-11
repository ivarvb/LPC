#include "BoundariesExtractor.h"

BoundariesExtractor::BoundariesExtractor()
{

}

void BoundariesExtractor::SetInputDatasetPath(const std::string& dataSetPath)
{
    InputDatasetPath = (*dataSetPath.rbegin() == '/') ? dataSetPath.substr(0, dataSetPath.length()-1) : dataSetPath;
}

void BoundariesExtractor::SetOutputDatasetPath(const std::string& outputPath)
{
    //delete '/' at the end of the path
    OutputDatasetPath = ( (*outputPath.rbegin()) == '/') ? outputPath.substr(0, outputPath.length()-1) : outputPath;
}


void BoundariesExtractor::ConnectBackground(GrayImageP& grayImage)
{
    const auto upperIndex = grayImage->GetRequestedRegion().GetUpperIndex();

    for (unsigned col=0; col <= upperIndex[0]; ++col)
    {
        grayImage->SetPixel({col,0}, Background);
        grayImage->SetPixel({col,upperIndex[1]}, Background);
    }


    for (unsigned row=0; row <= upperIndex[1]; ++row)
    {
        grayImage->SetPixel({0,row}, Background);
        grayImage->SetPixel({upperIndex[0], row}, Background);
    }


}


BoundariesExtractor::GrayImageP BoundariesExtractor::ConnectForeground(GrayImageP binaryImage)
{

    //Connecting background
    ConnectBackground(binaryImage);

    //Detecting background, thi goal is to set internal pixel to 0
    using ConnectedFilterType = itk::ConnectedThresholdImageFilter<GrayImageT, GrayImageT>;
    ConnectedFilterType::Pointer connectedThreshold = ConnectedFilterType::New();
    connectedThreshold->SetInput(binaryImage);
    connectedThreshold->SetLower(Background);
    connectedThreshold->SetUpper(Background);
    connectedThreshold->SetSeed({0,0});
    connectedThreshold->SetReplaceValue(Foreground);

    connectedThreshold->Update();

    return connectedThreshold->GetOutput();

}

BoundariesExtractor::GrayImageP BoundariesExtractor::ExtractBoundaries(GrayImageP connectedForeground)
{

/*  //Connecting background
    ConnectBackground(binaryImage);

    //Detecting background, thi goal is to set internal pixel to 0
    using ConnectedFilterType = itk::ConnectedThresholdImageFilter<GrayImageT, GrayImageT>;
    ConnectedFilterType::Pointer connectedThreshold = ConnectedFilterType::New();
    connectedThreshold->SetInput(binaryImage);
    connectedThreshold->SetLower(Background);
    connectedThreshold->SetUpper(Background);
    connectedThreshold->SetSeed({0,0});
    connectedThreshold->SetReplaceValue(Foreground);

    connectedThreshold->Update();
*/
    //VTKViewer::visualize<GrayImageT>(connectedThreshold->GetOutput());

    using binaryContourImageFilterType = itk::BinaryContourImageFilter<GrayImageT, GrayImageT>;
    binaryContourImageFilterType::Pointer binaryContourFilter = binaryContourImageFilterType::New();
    binaryContourFilter->SetInput(connectedForeground);
    binaryContourFilter->SetBackgroundValue(Background);
    binaryContourFilter->SetForegroundValue(Foreground);
    binaryContourFilter->FullyConnectedOn();
    binaryContourFilter->Update();

    io::printOK("Extract boundaries");


    return binaryContourFilter->GetOutput();

}

BoundariesExtractor::GrayImageP BoundariesExtractor::ThinningBoundaries(GrayImageP boundaries)
{

    using BinaryThinningImageFilterType = itk::BinaryThinningImageFilter<GrayImageT, GrayImageT>;
    BinaryThinningImageFilterType::Pointer binaryThinningImageFilter = BinaryThinningImageFilterType::New();
    binaryThinningImageFilter->SetInput(boundaries);
    binaryThinningImageFilter->Update();

    // Rescale the image so that it can be seen (the output is 0 and 1, we want 0 and 255)
    using RescaleType = itk::RescaleIntensityImageFilter<GrayImageT, GrayImageT>;
    RescaleType::Pointer rescaler = RescaleType::New();
    rescaler->SetInput(binaryThinningImageFilter->GetOutput());
    rescaler->SetOutputMinimum(Background);
    rescaler->SetOutputMaximum(Foreground);
    rescaler->Update();


    io::printOK("Thinning Boundaries");


    return rescaler->GetOutput();



}

BoundariesExtractor::GrayImageP BoundariesExtractor::GrayToBinary(GrayImageP grayImage, bool show)
{

    auto binaryImage = GrayImageT::New();
    binaryImage->SetRegions(grayImage->GetRequestedRegion());
    binaryImage->Allocate();


    itk::ImageRegionConstIterator<GrayImageT> gIt(grayImage, grayImage->GetRequestedRegion()); //edges it
    itk::ImageRegionIterator<GrayImageT>      bIt(binaryImage, binaryImage->GetRequestedRegion()); //output it


    for(; !gIt.IsAtEnd(); ++gIt, ++bIt)
    {
        //std::cout<<gIt.Get()<<std::endl;
        //bIt.Set( (gIt.Get() == Foreground )? Background : Foreground );

        bIt.Set( (gIt.Get() >= Foreground - 1 )? Background: Foreground);
    }

    if(show)
    {
        /* VTKViewer::visualize<GrayImageT>(binaryImage, "Gray to Binary"); */
    }


    io::printOK("Simple gray to binary");

    return binaryImage;

}


void BoundariesExtractor::SetGaussSigma(float sigma)
{
    GaussSigma = sigma;
}

BoundariesExtractor::GrayImageP BoundariesExtractor::LabelMapToBinaryImage(const LabelMapP& labelMap)
{

    //Label-map to RGB image
    using  rgbImageT =  itk::Image<itk::RGBPixel<unsigned char>, 2>;
    typedef itk::LabelMapToRGBImageFilter<LabelMapT, rgbImageT> RGBFilterType;
    typename RGBFilterType::Pointer labelMapToRGBFilter = RGBFilterType::New();
    labelMapToRGBFilter->SetInput(labelMap);
    labelMapToRGBFilter->Update();


    using rgbToGrayFilterT = itk::RGBToLuminanceImageFilter<rgbImageT, GrayImageT>;
    typename rgbToGrayFilterT::Pointer rgbToGrayFilter = rgbToGrayFilterT::New();
    rgbToGrayFilter->SetInput(labelMapToRGBFilter->GetOutput());
    rgbToGrayFilter->Update();


    using FilterType = itk::BinaryThresholdImageFilter<GrayImageT, GrayImageT>;
    typename FilterType::Pointer filter = FilterType::New();
    filter->SetInput(rgbToGrayFilter->GetOutput());
    filter->SetLowerThreshold(0);
    filter->SetUpperThreshold(1); //all values greater than black->0
    filter->SetOutsideValue(Foreground);
    filter->SetInsideValue(Background);
    filter->Update();

    return filter->GetOutput();



}


BoundariesExtractor::GrayImageP BoundariesExtractor::DeleteSmallComponents(GrayImageP edgesImage)
{

    using ConnectedComponentImageFilterType = itk::ConnectedComponentImageFilter<GrayImageT, GrayImageT>;
    ConnectedComponentImageFilterType::Pointer connected = ConnectedComponentImageFilterType::New();
    connected->SetInput(edgesImage);
    connected->FullyConnectedOn();
    connected->SetBackgroundValue(Background);
    connected->Update();


    using LabelImageToLabelMapFilterType =  itk::LabelImageToShapeLabelMapFilter<GrayImageT, LabelMapT>;
    typename LabelImageToLabelMapFilterType::Pointer labelImageToLabelMapFilter = LabelImageToLabelMapFilterType::New();
    labelImageToLabelMapFilter->SetInput(connected->GetOutput());
    labelImageToLabelMapFilter->Update();
    auto labelMap = labelImageToLabelMapFilter->GetOutput();

    if(ComponentsThreshold > 0)
    {
        std::vector<ShapeLabelObjectType::Pointer> labelsToRemove;
        for(unsigned i=1; i < labelMap->GetNumberOfLabelObjects(); ++i) //it starts in 1 because 0 is the background label
        {



            if(labelMap->GetLabelObject(i)->Size() <  ComponentsThreshold)
            {
                labelsToRemove.push_back(labelMap->GetLabelObject(i));
                //labelsToRemove.push_back(i);
            }

        }

        for(auto i : labelsToRemove)
        {
            //labelMap->RemoveLabel(i);
            labelMap->RemoveLabelObject(i);
        }
    }



    return  LabelMapToBinaryImage(labelMap);




}

void BoundariesExtractor::SetThinBoundariesOn()
{
    ThinBoundaries = true;
}

void BoundariesExtractor::SetThinBoundariesOff()
{
    ThinBoundaries = false;
}

BoundariesExtractor::GrayImageP BoundariesExtractor::GaussianBlur(const GrayImageP& inputImage, bool show)
{

    using FilterType = itk::SmoothingRecursiveGaussianImageFilter<GrayImageT, GrayImageT>;
    FilterType::Pointer smoothFilter = FilterType::New();
    smoothFilter->SetSigma(GaussSigma);
    smoothFilter->SetInput(inputImage);
    smoothFilter->Update();

    if(show)
    {

        /* VTKViewer::visualize<GrayImageT>(smoothFilter->GetOutput(), "Gaussian Blur"); */
    }

    io::printOK("Gaussian blur");
    return smoothFilter->GetOutput();

}

void BoundariesExtractor::SetSmallComponentsThreshold(std::uint16_t threshold)
{

    ComponentsThreshold = threshold;
}


void BoundariesExtractor::SetOutputMaskPath(const std::string& outputPath)
{
    this->OutputMaskPath = outputPath;
}


void BoundariesExtractor::Process(bool returnResults)
{

    if(InputDatasetPath==OutputDatasetPath)
    {
        std::cerr<<"Input Dataset path and Output Dataset Path MUST be different"<<std::endl;
        return;
    }

    //get all files in dataset path
    std::vector<std::string> imagePaths;
    std::vector<std::string> imageNames;
    // std::string imageName;
    for(auto filePath: std::filesystem::directory_iterator(InputDatasetPath))
    {
        imagePaths.push_back(filePath.path());
        imageNames.push_back( *io::Split( *io::Split(filePath.path(), '/').rbegin(), '.').begin() );

    }

    auto imagePathIt = imagePaths.begin();
    auto imageNameIt = imageNames.begin();
    GrayImageP boundaries;
    GrayImageP binaryFiltered;
    GrayImageP binaryImage;
    GrayImageP blur;
    GrayImageP connectedForeground;

    for(; imagePathIt != imagePaths.end(); ++imagePathIt, ++imageNameIt)
    {

        io::printWait("Boundary Extraction image "+*imageNameIt);
        //TODO consider smoothing the image

        blur = GaussianBlur(io::ReadImage<GrayImageT>(*imagePathIt), false);


        binaryImage = GrayToBinary(blur, false);


        binaryFiltered = DeleteSmallComponents(binaryImage);

        connectedForeground = ConnectForeground(binaryFiltered);

        boundaries = ExtractBoundaries(connectedForeground);

        //VTKViewer::visualize<GrayImageT>(boundaries);

        if(ThinBoundaries)
        {
            auto thinBoundaries = ThinningBoundaries(boundaries);

            if(OutputDatasetPath != "")
            {
                io::WriteImage<GrayImageT>(thinBoundaries, OutputDatasetPath+"/"+*imageNameIt+".tiff");
            }
            if(OutputMaskPath != "")
            {
                io::WriteImage<GrayImageT>(connectedForeground, OutputMaskPath+"/"+*imageNameIt+".tiff");
            }
            //VTKViewer::visualize<GrayImageT>(thinBoundaries);
            if(returnResults)
            {
                OutputImages.push_back(thinBoundaries);
            }
        }
        else
        {
            if(OutputDatasetPath != "")
            {
                io::WriteImage<GrayImageT>(boundaries, OutputDatasetPath+"/"+*imageNameIt+".tiff");
            }
            if(OutputMaskPath != "")
            {
                io::WriteImage<GrayImageT>(connectedForeground, OutputMaskPath+"/"+*imageNameIt+".tiff");
            }

            if(returnResults)
            {
                OutputImages.push_back(boundaries);
            }
        }


    }


}


