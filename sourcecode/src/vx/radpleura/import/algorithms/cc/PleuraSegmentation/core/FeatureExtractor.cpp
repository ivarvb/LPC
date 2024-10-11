#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor()
{

}

std::string FeatureExtractor::deleteSlash(std::string const& path)
{
    return (*path.rbegin() == '/') ? path.substr(0, path.length()-1) : path;
}


/*
Boundaries muts be binary images with background value equals to zero and foreground 255
*/
void FeatureExtractor::SetBoundariesPath(const std::string& boundariesPath)
{
    //delet last "/"
    BoundariesPath = (*boundariesPath.rbegin() == '/') ? boundariesPath.substr(0, boundariesPath.length()-1) : boundariesPath;
}

void FeatureExtractor::SetPleuraMasksPath(const std::string& path)
{
    PleuraMasksPath = (*path.rbegin() == '/') ? path.substr(0, path.length()-1) : path;
}

void FeatureExtractor::SetImagesPath(const std::string& imagesPath)
{
    ImagesPath = (*imagesPath.rbegin() == '/') ? imagesPath.substr(0, imagesPath.length()-1) : imagesPath;

}

void FeatureExtractor::SetLabelsPath(const std::string& labelsPath)
{

    LabelsPath = (*labelsPath.rbegin() == '/') ? labelsPath.substr(0, labelsPath.length()-1) : labelsPath;
}


void FeatureExtractor::SetKernelSize(const std::size_t& kernelSize)
{

    //kernel size must be odd
    this->KernelSize = (kernelSize%2==0) ? kernelSize+1 : kernelSize;
}


void FeatureExtractor::FindCenters2(GrayImageP boundaries, CentersT& centers)
{

/*
    //tmp image for testing
    auto tmpImage = GrayImageT::New();
    tmpImage->SetRegions(boundaries->GetRequestedRegion());
    tmpImage->Allocate();
    tmpImage->FillBuffer(210);
*/
    centers.clear();
    using indexT = GrayImageT::IndexType;


    const auto& distance =  math::squaredEuclideanDistance<indexT, 2, unsigned>; //avoid squared root computation
    auto squaredKernelSize   = KernelSize*KernelSize;

    //connected components
    using ConnectedComponentImageT = itk::ConnectedComponentImageFilter<GrayImageT, GrayImageT>;
    ConnectedComponentImageT::Pointer connectedComponentsFilter = ConnectedComponentImageT::New();
    connectedComponentsFilter->SetInput(boundaries);
    connectedComponentsFilter->FullyConnectedOn();
    connectedComponentsFilter->SetBackgroundValue(Background);
    connectedComponentsFilter->Update();
    auto connectedComponents = connectedComponentsFilter->GetOutput();



   /* using RGBFilterType = itk::LabelToRGBImageFilter<GrayImageT, RGBImageT>;
    RGBFilterType::Pointer rgbFilter = RGBFilterType::New();
    rgbFilter->SetInput(connectedComponents);
    rgbFilter->Update();
    VTKViewer::visualize<RGBImageT>(rgbFilter->GetOutput(), "Centers");
    */

    itk::ImageRegionConstIterator<GrayImageT> compIt(connectedComponents, connectedComponents->GetLargestPossibleRegion());

    std::vector<GrayImageT::PixelType> labelsAux;


    unsigned nObjects = connectedComponentsFilter->GetObjectCount();
    for (compIt.GoToBegin(); !compIt.IsAtEnd(); ++compIt)
    {
        if(compIt.Get()==Background)
        {
            continue;
        }
        if ( std::find( labelsAux.begin(), labelsAux.end(), compIt.Get()) == labelsAux.end())
        {
            centers.push_back(compIt.GetIndex());
            labelsAux.push_back(compIt.Get());
            //tmpImage->SetPixel(compIt.GetIndex(), 0);

        }
        if(labelsAux.size() == nObjects)
        {
            break;
        }

    }

    for (compIt.GoToBegin(); !compIt.IsAtEnd(); ++compIt)
    {

        const auto& label = compIt.Get();
        //Ignore background pixels
        if(label == Background)
        {
            //tmpImage->SetPixel(compIt.GetIndex(), 255);
            continue;
        }

        const auto& index = compIt.GetIndex();

        bool insertflag=true;

        //verify distances
        auto centIt = centers.begin();
        auto labelIt = labelsAux.begin();
        for (;  centIt != centers.end(); ++centIt, ++labelIt)
        {

            if(label == *labelIt)//same component
            {
                if(distance(index, *centIt) < squaredKernelSize)
                {
                    insertflag=false;
                    break;
                }
            }

        }

        if(insertflag)
        {
            centers.push_back(index);
            labelsAux.push_back(label);
            //tmpImage->SetPixel(index, 0);
        }



    }

    //VTKViewer::visualize<GrayImageT>(tmpImage, "Centers");
    io::printOK("Compute Centers");



}


/*


*/
void FeatureExtractor::FindCenters(GrayImageP boundaries, std::vector<GrayImageT::IndexType>& centers)
{


    using DuplicatorType = itk::ImageDuplicator<GrayImageT>;
    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(boundaries);
    duplicator->Update();
    auto boundariesDuplicate = duplicator->GetOutput();

    /* //tmp image for testing
    auto tmpImage = GrayImageT::New();
    tmpImage->SetRegions(boundaries->GetRequestedRegion());
    tmpImage->Allocate();
    tmpImage->FillBuffer(100); //100 is just a middle-value different enough from 0 and 255
*/
    itk::NeighborhoodIterator<GrayImageT>::RadiusType radius;
    radius.Fill(KernelSize/2);

    itk::NeighborhoodIterator<GrayImageT> it(radius, boundariesDuplicate, boundariesDuplicate->GetRequestedRegion());
    //ensure centers vector is empty
    centers.clear();

    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {

        if(it.GetCenterPixel() == Foreground)
        {
            centers.push_back(it.GetIndex());

            //for (auto nIt = it.Begin(); nIt != it.End() ; ++nIt)
            for (unsigned i=0; i < it.Size(); ++i)
            {
                //cout << typeid(nIt).name() << '\n';
                //(**nIt) = Background;
                //std::cout<<(**nIt)<<std::endl;
                const auto& index = it.GetIndex(i);
                if(boundariesDuplicate->GetRequestedRegion().IsInside(index))
                {
                    boundariesDuplicate->SetPixel(index, Background);
                    //tmpImage->SetPixel(index, Foreground);
                }
            }
            //VTKViewer::visualize<GrayImageT>(boundariesDuplicate, "Centers");
        }


    }

    //VTKViewer::visualize<GrayImageT>(tmpImage, "Centers");
    io::printOK("Compute Centers");

}

void FeatureExtractor::ComputeLBPFeatures(GrayImageP grayImage,
                                          GrayImageP masks,
                                          const CentersT& centers,
                                          std::vector<std::vector<unsigned long>>& lbpHistograms)
{
    lbpHistograms.clear();
    lbpHistograms.reserve(centers.size());

    using grayPixelT = unsigned;
    using dlibGrayImageT = dlib::array2d<grayPixelT>;

    //ITK to Dlib image
    dlibGrayImageT dlibGrayImage;
    image::ITKToDlib<grayPixelT, grayPixelT>(grayImage, dlibGrayImage);

    dlibGrayImageT lbpImage;
    dlib::make_uniform_lbp_image<dlibGrayImageT, dlibGrayImageT>(dlibGrayImage, lbpImage);

    dlibGrayImageT dlibMasks;
    image::ITKToDlib<grayPixelT, grayPixelT>(masks, dlibMasks);

    // dlib::image_window my_window(dlibMasks, "LBP");
    //my_window.wait_until_closed();

    const auto& ExtractNeighborhoodDlib = image::ExtractNeighborhoodDLib<dlibGrayImageT, GrayImageT::IndexType>;
    const auto& Histogram = image::ComputeHistogram<dlibGrayImageT, std::vector<unsigned long> >;


    for(const auto& index: centers)
    {
        FeaturesT features(59,1); //59 LBP
        dlibGrayImageT imgTile, maskTile;

        ExtractNeighborhoodDlib(lbpImage, index, KernelSize, imgTile);
        ExtractNeighborhoodDlib(dlibMasks, index, KernelSize, maskTile);

        /*
        dlib::image_window my_window2(imgTile, "LBP");
        my_window2.wait_until_closed();

        dlib::image_window my_window3(maskTile, "Mask");
        my_window3.wait_until_closed();

*/
        // dlib::get_histogram(neighborhood, lbpHistogramLongT, 59);

        //dlib::array<unsigned long> lbpHistogram(59);
        std::vector<unsigned long> lbpHistogram(59,0);
        Histogram(imgTile, maskTile, lbpHistogram);
        lbpHistograms.emplace_back(lbpHistogram);

    }

    io::printOK("Local Binary Pattern");


}

/*
void FeatureExtractor::ComputeShapeFeatures(GrayImageP grayImage, const CentersT& centers, FeaturesVectorT& featuresVector)
{

    //label maps types
    using LabelType = unsigned short;
    using OutputImageT = itk::Image<unsigned, 2>;
    using ShapeLabelObjectType = itk::ShapeLabelObject<LabelType, 2>;
    using LabelMapType = itk::LabelMap<ShapeLabelObjectType>;

    using ConnectedComponentImageT = itk::ConnectedComponentImageFilter<GrayImageT, OutputImageT>;
    using LabelImageToShapeLabelMapT = itk::LabelImageToShapeLabelMapFilter<OutputImageT, LabelMapType>;
    using RelabelComponentT = itk::RelabelComponentImageFilter<OutputImageT, OutputImageT>;


    const auto& extractNeighborhood = image::ExtractNeighborhoodITK<GrayImageT>;

    featuresVector.clear();
    featuresVector.reserve(centers.size());

    FeaturesT features(3,1);

    for(const auto& center: centers)
    {
        GrayImageT::Pointer tile;
        extractNeighborhood(grayImage, center, KernelSize, tile);

        //VTKViewer::visualize<GrayImageT>(tile, "neigh");


        ConnectedComponentImageT::Pointer connected = ConnectedComponentImageT::New();
        connected->SetInput(tile);
        connected->FullyConnectedOn();
        connected->SetBackgroundValue(Background); //black
        connected->Update();

        //Relabel objects so that the largest object has label #1,
        RelabelComponentT::Pointer relabelFilter = RelabelComponentT::New();
        relabelFilter->SetInput(connected->GetOutput());
        relabelFilter->Update();

        LabelImageToShapeLabelMapT::Pointer labelImageToShapeMap = LabelImageToShapeLabelMapT::New();
        labelImageToShapeMap->SetInput(relabelFilter->GetOutput());
        labelImageToShapeMap->Update();

        auto largestObject = labelImageToShapeMap->GetOutput()->GetLabelObject(1);


        //fractal dimension
        auto fractalDimension = std::make_unique<FractalDimensionCalculator<GrayImageT>>();
        fractalDimension->SetInputImage(tile);
        fractalDimension->PrintWarningsOff();
        fractalDimension->SetUnitTileLenght(2);
        fractalDimension->Compute();

        features(0,0) = largestObject->GetFlatness();
        features(1,0) = largestObject->GetRoundness();
        features(2,0) = fractalDimension->GetDimension();
        featuresVector.emplace_back(features);

        //std::cout<<features<<std::endl;


        
        ////Label-map to RGB image
        //using  rgbImageT =  itk::Image<itk::RGBPixel<unsigned char>, 2>;
        //typedef itk::LabelMapToRGBImageFilter<LabelMapType, rgbImageT> RGBFilterType;
        //typename RGBFilterType::Pointer labelMapToRGBFilter = RGBFilterType::New();
        //labelMapToRGBFilter->SetInput(labelImageToShapeMap->GetOutput());
        //labelMapToRGBFilter->Update();



        //VTKViewer::visualize<rgbImageT>(labelMapToRGBFilter->GetOutput(), "Connected components RGB");


    }
    io::printOK("Shape features");
}
*/


/*
void FeatureExtractor::FractalDimensionBoxCounting(const GrayImageP& grayImage, const CentersT& centers, std::vector<double>& fractalDimensions)
{

    //VTKViewer::visualize<GrayImageT>(grayImage, "neigh");

    const auto& extractNeighborhood = image::ExtractNeighborhoodITK<GrayImageT>;

    fractalDimensions.clear();
    fractalDimensions.reserve(centers.size());

    for(const auto& center: centers)
    {
        GrayImageT::Pointer neighborhood;
        extractNeighborhood(grayImage, center, KernelSize, neighborhood);
        //VTKViewer::visualize<GrayImageT>(neighborhood, "Fractal dim kernel");

        FractalDimensionCalculator<GrayImageT> fractalDimension;
        fractalDimension.PrintWarningsOff();
        fractalDimension.SetBackGround(Background);
        fractalDimension.SetInputImage(neighborhood);
        fractalDimension.Compute();
        fractalDimensions.emplace_back(fractalDimension.GetDimension());

    }

    io::printOK("Fractal dimension boundaries");



}
*/


/*
void FeatureExtractor::StochastichFractalFeatures(GrayImageP grayImage, GrayImageP masks, const CentersT& centers, FeaturesVectorT& featuresVector)
{


    //label maps types
    using LabelType = unsigned short;
    using OutputImageT = itk::Image<unsigned, 2>;
    using ShapeLabelObjectType = itk::ShapeLabelObject<LabelType, 2>;
    using LabelMapType = itk::LabelMap<ShapeLabelObjectType>;

    const auto& extractNeighborhood = image::ExtractNeighborhoodITK<GrayImageT>;

    featuresVector.clear();
    featuresVector.reserve(centers.size());

    using  StochasticFractalFilter =  itk::StochasticFractalDimensionImageFilter<GrayImageT, GrayImageT, GrayImageT>;

    FeaturesT features(1,1);

    for(const auto& center: centers)
    {
        //fractal image
        GrayImageT::Pointer tileImage;
        GrayImageT::Pointer mask;

        extractNeighborhood(grayImage, center, KernelSize, tileImage);
        extractNeighborhood(masks, center, KernelSize, mask);

        //VTKViewer::visualize<GrayImageT>(tileImage,"Tile");
        // VTKViewer::visualize<GrayImageT>(mask,"tile mask"); 


        StochasticFractalFilter::Pointer stochasticFractalPointer = StochasticFractalFilter::New();
        stochasticFractalPointer->SetInput(tileImage);
        stochasticFractalPointer->SetMaskImage(mask);
        stochasticFractalPointer->Update();


        //stochasticFractalPointer->get
        auto fractalOutput = stochasticFractalPointer->GetOutput();


        using FilterType = itk::RescaleIntensityImageFilter<GrayImageT, GrayImageT>;
        FilterType::Pointer filter = FilterType::New();
        filter->SetInput(fractalOutput);
        filter->SetOutputMinimum(0);
        filter->SetOutputMaximum(255);
        filter->Update();


        // VTKViewer::visualize<GrayImageT>(tileImage,"Tile");
        // VTKViewer::visualize<GrayImageT>(filter->GetOutput(),"fractal"); 

    }

    io::printOK("Fractal and Shape dimension features");



}
*/


void FeatureExtractor::ComputeCooccurrenceMatrixFeatures(GrayImageP grayImage,
                                                         GrayImageP maskImage,
                                                         const CentersT& centers,
                                                         std::vector<std::vector<double>> &featuresVector)
{

    featuresVector.clear();
    featuresVector.reserve(centers.size());

    using ScalarImageToCooccurrenceMatrixFilter =  itk::Statistics::ScalarImageToCooccurrenceMatrixFilter<GrayImageT>;

    using histogramT =  ScalarImageToCooccurrenceMatrixFilter::HistogramType;
    using offsetVectorT = itk::VectorContainer<unsigned char, ScalarImageToCooccurrenceMatrixFilter::OffsetType>;

    //using offsetT = GrayImageT::OffsetType;
    //offsetT offset = {-1,1};FindCenters

    offsetVectorT::Pointer offsets = offsetVectorT::New();
    // offsets->reserve(2);

    offsets->push_back({-1,1});
    offsets->push_back({1,1});
    offsets->push_back({1,-1});


    using  histToFeaturesT = itk::Statistics::HistogramToTextureFeaturesFilter<histogramT>;


    const auto extractNeighborhood = image::ExtractNeighborhoodITK<GrayImageT>;

    featuresVector.clear();
    featuresVector.reserve(centers.size());

    for(auto it = centers.begin(); it != centers.end(); ++it)
    {

        GrayImageT::Pointer tile;
        GrayImageT::Pointer tileMask;

        extractNeighborhood(grayImage, *it, KernelSize, tile);
        extractNeighborhood(maskImage, *it, KernelSize, tileMask);

        //std::cout<<*it<<std::endl;
        //VTKViewer::visualize<GrayImageT>(tile, "neigh");
        //VTKViewer::visualize<GrayImageT>(tileMask, "neigh");

        ScalarImageToCooccurrenceMatrixFilter::Pointer scalarImageToCooccurrenceMatrixFilter = ScalarImageToCooccurrenceMatrixFilter::New();

        scalarImageToCooccurrenceMatrixFilter->SetInput(tile);
        scalarImageToCooccurrenceMatrixFilter->SetOffsets(offsets);
        scalarImageToCooccurrenceMatrixFilter->SetNumberOfBinsPerAxis(256);
        scalarImageToCooccurrenceMatrixFilter->SetPixelValueMinMax(0,255);
        scalarImageToCooccurrenceMatrixFilter->SetMaskImage(tileMask);
        scalarImageToCooccurrenceMatrixFilter->SetInsidePixelValue(255);
        scalarImageToCooccurrenceMatrixFilter->Update();


        histToFeaturesT::Pointer histFeaturesFilter =  histToFeaturesT::New();
        histFeaturesFilter->SetInput(scalarImageToCooccurrenceMatrixFilter->GetOutput());
        histFeaturesFilter->Update();

        std::vector<double> histFeatures(4);
        histFeatures[0] = histFeaturesFilter->GetEnergy();
        histFeatures[1] = histFeaturesFilter->GetEntropy();
        histFeatures[2] = histFeaturesFilter->GetCorrelation();
        histFeatures[3] = histFeaturesFilter->GetInertia();

        //std::cout<<histFeatures[0]<<std::endl;

        featuresVector.emplace_back(histFeatures);

        //std::cout<<features<<std::endl;

    }

    io::printOK("Co-occurrence Matrix features");

}

void FeatureExtractor::WriteFeaturesCSV(const std::string& fileName, bool writeHeader) const
{
    std::ofstream csvFile(fileName);

    if(writeHeader)
    {
        csvFile<<"Image_Name,Col,Row,";
        for(unsigned i=0; i< FeaturesVector.begin()->size() ; ++i  )
        {
            csvFile<<"feature_"<<i+1<<",";
        }
        csvFile<<"Label"<<std::endl;

    }

    auto fIt = FeaturesVector.begin();
    auto lIt = LabelsVector.begin();
    auto cIt = CentersVector.begin();

    auto iIt = ImagesNames.begin();
    auto sNrIt = CentersNumberPerImage.begin();

    std::uint16_t samplesCounter = 0;
    while(fIt != FeaturesVector.end())
    {
        //writing center index
        //std::cout<<(*iIt)<<std::endl;

        //Writing Image names
        csvFile.write((*iIt).c_str(), (*iIt).length());
        csvFile<<",";

        //Center index
        csvFile<<(*cIt)[0]<<","<<(*cIt)[1]<<",";

        //Features
        for(const auto& feature : *fIt  )
        {
            csvFile<<feature<<",";
        }

        //Label
        csvFile<<*lIt<<std::endl;

        ++fIt;
        ++lIt;
        ++cIt;
        if(++samplesCounter == *sNrIt )
        {
            ++iIt;
            ++sNrIt;
            samplesCounter = 0;
        }


    }

    csvFile.close();
    io::printOK("Writing CSV file");
}

void FeatureExtractor::FindLabels(const RGBImageP& labelsImage,
                                  const CentersT& centersVector,
                                  std::vector<double>& labelsVector)
{


    labelsVector.clear();
    labelsVector.reserve(centersVector.size());

    itk::RGBPixel<unsigned char> green;
    green.Set(4, 255, 0);

    for (const auto& center : centersVector )
    {
        labelsVector.push_back(( labelsImage->GetPixel(center) == green )? Pleura : NonPleura );
    }

    io::printOK("Finding labels");


}

void FeatureExtractor::SetMasksPath(std::string const& masksPath)
{
    MasksPath = FeatureExtractor::deleteSlash(masksPath);
}



FeatureExtractor::GrayImageP FeatureExtractor::ScaleMaskObjectsSize(GrayImageP mask)
{

    auto size  = mask->GetRequestedRegion().GetSize();
    auto spacing = mask->GetSpacing();
    itk::Index<2> centralPixel;
    centralPixel[0] = size[0] / 2;
    centralPixel[1] = size[1] / 2;

    itk::Point<double, 2> centralPoint;
    centralPoint[0] = centralPixel[0];
    centralPoint[1] = centralPixel[1];

    using ScaleTransformType = itk::ScaleTransform<double, 2>;
    ScaleTransformType::Pointer scaleTransform = ScaleTransformType::New();
    ScaleTransformType::ParametersType parameters = scaleTransform->GetParameters();
    parameters[1] = MaskScaleFactor;//1.5;
    parameters[0] = MaskScaleFactor;//1.5;

    scaleTransform->SetParameters(parameters);
    scaleTransform->SetCenter(centralPoint);

    using LinearInterpolatorType = itk::LinearInterpolateImageFunction<GrayImageT, double>;
    LinearInterpolatorType::Pointer interpolator = LinearInterpolatorType::New();

    using ResampleFilterType = itk::ResampleImageFilter<GrayImageT, GrayImageT>;
    ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

    using InvertIntensityImageFilterType = itk::InvertIntensityImageFilter<GrayImageT>;
    InvertIntensityImageFilterType::Pointer invertIntensityFilter = InvertIntensityImageFilterType::New();
    invertIntensityFilter->SetMaximum(Foreground);
    invertIntensityFilter->SetInput(mask);

    resampleFilter->SetInput(invertIntensityFilter->GetOutput());
    resampleFilter->SetTransform(scaleTransform);
    resampleFilter->SetInterpolator(interpolator);
    resampleFilter->SetSize(size);
    resampleFilter->SetOutputSpacing(spacing);


    resampleFilter->Update();

    invertIntensityFilter = InvertIntensityImageFilterType::New();
    invertIntensityFilter->SetMaximum(Foreground);
    invertIntensityFilter->SetInput(resampleFilter->GetOutput());
    invertIntensityFilter->Update();


    auto result = invertIntensityFilter->GetOutput();
    result->SetRegions(mask->GetRequestedRegion());
    result->Update();
    //VTKViewer::visualize<GrayImageT>(result, "scaled");
    return result;





}

FeatureExtractor::GrayImageP FeatureExtractor::MatchMaskObjecst(GrayImageP maskOriginal, GrayImageP maskScaled)
{
    //label maps types
    using LabelType = unsigned;
    //using OutputImageT = itk::Image<unsigned, 2>;

    using ShapeLabelObjectType = itk::ShapeLabelObject<LabelType, 2>;
    using LabelMapType = itk::LabelMap<ShapeLabelObjectType>;

    using ConnectedComponentImageT = itk::ConnectedComponentImageFilter<GrayImageT, GrayImageT>;
    using LabelImageToShapeLabelMapT = itk::LabelImageToShapeLabelMapFilter<GrayImageT, LabelMapType>;


    using InvertIntensityImageFilterType = itk::InvertIntensityImageFilter<GrayImageT>;
    using RelabelComponentT = itk::RelabelComponentImageFilter<GrayImageT, GrayImageT>;


    //compute original mask's centroids
    InvertIntensityImageFilterType::Pointer invertIntensityFilter = InvertIntensityImageFilterType::New();
    invertIntensityFilter->SetMaximum(Foreground);
    invertIntensityFilter->SetInput(maskOriginal);
    //invertIntensityFilter->Update();

    ConnectedComponentImageT::Pointer connectedOrg = ConnectedComponentImageT::New();
    connectedOrg->SetInput(invertIntensityFilter->GetOutput());
    connectedOrg->FullyConnectedOn();
    //connectedOrg->SetBackgroundValue(Background);
    //connectedOrg->Update();

    //Relabel objects so that the largest object has label #1,
    RelabelComponentT::Pointer relabelFilter = RelabelComponentT::New();
    relabelFilter->SetInput(connectedOrg->GetOutput());
    //relabelFilter->Update();

    LabelImageToShapeLabelMapT::Pointer labelImageToShapeMapOrg = LabelImageToShapeLabelMapT::New();
    labelImageToShapeMapOrg->SetInput(relabelFilter->GetOutput());
    labelImageToShapeMapOrg->Update();
    auto labelMapOrg = labelImageToShapeMapOrg->GetOutput(); //original

    /*
    //Label-map to RGB image
    using  rgbImageT =  itk::Image<itk::RGBPixel<unsigned char>, 2>;
    typedef itk::LabelMapToRGBImageFilter<LabelMapType, rgbImageT> RGBFilterType;
    typename RGBFilterType::Pointer labelMapToRGBFilter = RGBFilterType::New();
    labelMapToRGBFilter->SetInput(labelMapOrg);
    labelMapToRGBFilter->Update();
    VTKViewer::visualize<rgbImageT>(labelMapToRGBFilter->GetOutput(), "Mask org");
*/

    //compute scaled mask's
    InvertIntensityImageFilterType::Pointer invertIntensityFilterSca = InvertIntensityImageFilterType::New();
    invertIntensityFilterSca->SetMaximum(Foreground);
    invertIntensityFilterSca->SetInput(maskScaled);
    //invertIntensityFilterSca->Update();

    ConnectedComponentImageT::Pointer connectedSca = ConnectedComponentImageT::New();
    connectedSca->SetInput(invertIntensityFilterSca->GetOutput());
    connectedSca->FullyConnectedOn();
    //connectedSca->SetBackgroundValue(Background);
    //connectedSca->Update();


    RelabelComponentT::Pointer relabelFilterSca = RelabelComponentT::New();
    relabelFilterSca->SetInput(connectedSca->GetOutput());
    //relabelFilterSca->Update();

    LabelImageToShapeLabelMapT::Pointer  labelImageToShapeMapSca = LabelImageToShapeLabelMapT::New();
    labelImageToShapeMapSca->SetInput(relabelFilterSca->GetOutput());
    labelImageToShapeMapSca->Update();
    auto labelMapSca = labelImageToShapeMapSca->GetOutput(); //scaled

    /*
    //Label-map to RGB image
    using  rgbImageT =  itk::Image<itk::RGBPixel<unsigned char>, 2>;
    typedef itk::LabelMapToRGBImageFilter<LabelMapType, rgbImageT> RGBFilterType;
    typename RGBFilterType::Pointer labelMapToRGBFilter = RGBFilterType::New();
    labelMapToRGBFilter->SetInput(labelMapSca);
    labelMapToRGBFilter->Update();
    VTKViewer::visualize<rgbImageT>(labelMapToRGBFilter->GetOutput(), "Mask org");
    */


    //std::cout<<labelMapOrg->GetNumberOfLabelObjects()<<std::endl;
    //std::cout<<labelMapSca->GetNumberOfLabelObjects()<<std::endl;


    //Create result image
    GrayImageP maskScaledMatch = GrayImageT::New();
    maskScaledMatch->SetRegions(maskOriginal->GetRequestedRegion());
    maskScaledMatch->Allocate();
    maskScaledMatch->FillBuffer(Foreground);

    LabelMapType::ConstIterator labelMapScaIt(labelMapSca);
    LabelMapType::ConstIterator labelMapOrgIt(labelMapOrg);

    //for some reason the number of objecst is note the same after scaling
    while(!labelMapScaIt.IsAtEnd() && !labelMapOrgIt.IsAtEnd())
    {
        auto labelObjOrg = labelMapOrgIt.GetLabelObject();
        auto labelObjSca = labelMapScaIt.GetLabelObject();

        auto centObjOrg = labelObjOrg->GetCentroid();
        auto centObjSca = labelObjSca->GetCentroid();

        ShapeLabelObjectType::ConstIndexIterator labelObjIt(labelObjSca);

        while(!labelObjIt.IsAtEnd())
        {
            GrayImageT::IndexType indexSca = labelObjIt.GetIndex();

            GrayImageT::IndexType newPos;
            newPos[0] = (static_cast<int>(indexSca[0]) - centObjSca[0]) + centObjOrg[0];
            newPos[1] = (static_cast<int>(indexSca[1]) - centObjSca[1]) + centObjOrg[1];

            maskScaledMatch->SetPixel(newPos, Background);
            ++labelObjIt;
        }

        ++labelMapScaIt;
        ++labelMapOrgIt;
    }

    return maskScaledMatch;
    //VTKViewer::visualize<GrayImageT>(maskScaled, "Mask scaled");
    //VTKViewer::visualize<GrayImageT>(maskScaledMatch, "Mask scaled matched");

}

/**
mask must be  background = 0 and foreground = 255
*/
FeatureExtractor::GrayImageP FeatureExtractor::FindROIMask(GrayImageP maskOriginal, GrayImageP maskScaledCentered)
{

    //Create result image
    GrayImageP roiMask = GrayImageT::New();
    roiMask->SetRegions(maskOriginal->GetRequestedRegion());
    roiMask->Allocate();
    roiMask->FillBuffer(Foreground);

    itk::ImageRegionConstIterator<GrayImageT> orgIt(maskOriginal,  maskOriginal->GetRequestedRegion());
    itk::ImageRegionIterator     <GrayImageT> scaIt(maskScaledCentered, maskScaledCentered->GetRequestedRegion());
    itk::ImageRegionIterator     <GrayImageT> roiIt(roiMask, roiMask->GetRequestedRegion());

    for(; !orgIt.IsAtEnd() ; ++orgIt, ++scaIt, ++roiIt)
    {
        roiIt.Set( (scaIt.Get()+orgIt.Get() == Foreground )? Foreground : Background);
        // roiIt.Set( (scaIt.Get()+orgIt.Get() == Background )? Foreground : orgIt.Get());

    }

    //VTKViewer::visualize<GrayImageT>(maskOriginal, "Mask org");
    //VTKViewer::visualize<GrayImageT>(roiMask, "Pleura mask");


    return roiMask;



}

FeatureExtractor::GrayImageP FeatureExtractor::ErodeMask(GrayImageP maskOriginal)
{
    using StructuringElementType = itk::FlatStructuringElement<2>;
    StructuringElementType::RadiusType radius;
    radius.Fill(ErodeRadius);
    StructuringElementType structuringElement = StructuringElementType::Ball(radius);

    using BinaryErodeImageFilterType = itk::BinaryErodeImageFilter<GrayImageT, GrayImageT, StructuringElementType>;

    BinaryErodeImageFilterType::Pointer erodeFilter = BinaryErodeImageFilterType::New();
    erodeFilter->SetInput(maskOriginal);
    erodeFilter->SetKernel(structuringElement);
    erodeFilter->SetForegroundValue(0); // Intensity value to erode
    erodeFilter->SetBackgroundValue(255);   // Replacement value for eroded voxels
    erodeFilter->Update();

    //VTKViewer::visualize<GrayImageT>(erodeFilter->GetOutput(), "Mask roi");
    return erodeFilter->GetOutput();


}

void FeatureExtractor::CreateFeaturesVector(const std::vector< double>& fractal,
                                            const std::vector<std::vector<unsigned long>>& lbpHistograms,
                                            const std::vector<std::vector<double>>& cooccurrence,
                                            const std::vector<std::vector<double>>& moments,
                                            FeaturesVectorT& featuresVector)
{

    featuresVector.clear();
    featuresVector.reserve(fractal.size());

    //fractal =1
    unsigned featuresSize = 1 + lbpHistograms[0].size() + cooccurrence[0].size()+moments[0].size();
    FeaturesT features(featuresSize,1);

    for(unsigned i=0;i< fractal.size(); ++i)
    {
        unsigned fIndex=0;
        //Fractal
        features(fIndex++,0) = fractal[i];
        //LBP
        for(const auto lbp: lbpHistograms[i])
        {
            features(fIndex++, 0) = lbp;
        }

        //coo
        for(const auto coo: cooccurrence[i])
        {
            features(fIndex++,0) = coo;
        }

        //moments
        for(const auto mo: moments[i])
        {
            features(fIndex++,0) = mo;
        }

        featuresVector.emplace_back(features);

    }


    io::printOK("Creating features vector");

}

void FeatureExtractor::ComputeStatisticalMoments(const GrayImageP& image,
                                                 const GrayImageP& mask,
                                                 const CentersT& centers,
                                                 std::vector<std::vector<double>>& moments)
{

/*
    //Cast to char because itk moments filter mask works only with this type
    using ImageChar = itk::Image<unsigned char, 2>;
    using CastFilterType = itk::CastImageFilter<GrayImageT, ImageChar>;

    auto castFilter = CastFilterType::New();
    castFilter->SetInput(image);
    castFilter->Update();
    auto imageChar = castFilter->GetOutput();

    auto castFilterMask = CastFilterType::New();
    castFilterMask->SetInput(mask);
    castFilterMask->Update();
    auto maskChar = castFilterMask->GetOutput();

    //clear
    moments.clear();
    moments.reserve(centers.size());

    using MomentsCalculatorType =  itk::ImageMomentsCalculator<ImageChar>;
    using MaskType = itk::ImageMaskSpatialObject<2, ImageChar::PixelType>;

    const auto extractNeighborhood = image::ExtractNeighborhoodITK<ImageChar>;


    for(auto it = centers.begin(); it != centers.end(); ++it)
    {

        ImageChar::Pointer tile;
        ImageChar::Pointer tileMask;

        extractNeighborhood(imageChar, *it, KernelSize, tile);
        extractNeighborhood(maskChar, *it, KernelSize, tileMask);

        //VTKViewer::visualize<ImageChar>(tileMask, "Mask roi");

        MaskType::Pointer spatialObjectMask = MaskType::New();
        spatialObjectMask->SetImage(tileMask);
        spatialObjectMask->Update();

        MomentsCalculatorType::Pointer momentsCalculator = MomentsCalculatorType::New();
        momentsCalculator->SetImage(tile);
        momentsCalculator->SetSpatialObjectMask(spatialObjectMask);
        momentsCalculator->Compute();

        auto firstMomentsITK = momentsCalculator->GetFirstMoments();

        std::vector<double> tileMoments(firstMomentsITK.Size());

        util::VectorToVectorCast(firstMomentsITK, tileMoments, tileMoments.size());
        //io::PrintVector(tileMoments, tileMoments.size());

        moments.emplace_back(tileMoments);

    }
*/


}

void FeatureExtractor::SetErodeRadius(int radius)
{

    this->ErodeRadius = radius;
}

void FeatureExtractor::Process()
{

    //get all files in dataset path
    std::vector<std::string> imagesPaths;
    ImagesNames.clear();
    for(auto filePath: std::filesystem::directory_iterator(ImagesPath))
    {
        //imagesNames.push_back( *io::Split( *io::Split(filePath.path(), '/').rbegin(), '.').begin() );
        ImagesNames.push_back( *io::Split(filePath.path(), '/').rbegin() );
    }

    GrayImageP boundaries, image;
    RGBImageP labels;

    std::vector<GrayImageT::IndexType> centers;
    FeaturesVectorT featuresVector;
    std::vector<double> labelsVector;


    CentersNumberPerImage.clear();
    for(const auto& imageName : ImagesNames)
    {
        io::printWait("Feature extraction image "+imageName);

        image      = io::ReadImage<GrayImageT>(ImagesPath+"/"+imageName);
        boundaries = io::ReadImage<GrayImageT>(BoundariesPath+"/"+imageName);
        /* labels     = io::ReadImage<RGBImageT >(LabelsPath+"/"+imageName); */
        auto mask  = io::ReadImage<GrayImageT>(MasksPath+"/"+imageName);


/* 
        //centers
        //FindCenters(boundaries, centers);
        FindCenters2(boundaries, centers);
        FindLabels(labels, centers, labelsVector);
*/

        //ROI pleura-> segmentation
        auto erodedMask = ErodeMask(mask);
        auto pleuraMask = FindROIMask(mask, erodedMask);

        io::WriteImage<GrayImageT>(pleuraMask,  PleuraMasksPath+"/"+imageName);
        /* io::WriteImage<GrayImageT>(erodedMask,  PleuraMasksPath+"/"+imageName); */


/* 
        //fractal
        std::vector<double> fractalDimensions;
        FractalDimensionBoxCounting(boundaries, centers, fractalDimensions);


        //LBP
        std::vector<std::vector<unsigned long>> lbpHistograms;
        ComputeLBPFeatures(image, pleuraMask, centers, lbpHistograms);


        //Cooccurrence
        std::vector<std::vector<double>> cooHistograms;
        ComputeCooccurrenceMatrixFeatures(image, pleuraMask, centers, cooHistograms);


        //moments
        std::vector<std::vector<double>> moments;
        ComputeStatisticalMoments(image, pleuraMask, centers, moments);


        CreateFeaturesVector(fractalDimensions, lbpHistograms,  cooHistograms, moments, featuresVector);

        FeaturesVector.insert(FeaturesVector.end(), std::move_iterator(featuresVector.begin()),  std::move_iterator(featuresVector.end()));
        CentersVector.insert(CentersVector.end(),  std::move_iterator(centers.begin()), std::move_iterator(centers.end()));
        LabelsVector.insert(LabelsVector.end(), std::move_iterator(labelsVector.begin()), std::move_iterator(labelsVector.end()));

        CentersNumberPerImage.emplace_back(centers.size());
 */

    }


    io::printOK("Feature extraction for trainning");

}
