#include "ShowPrediction.h"

ShowPrediction::ShowPrediction()
{

}


void ShowPrediction::ReadCSV(const std::string& fileName,
                             unsigned imageNameIndex,
                             unsigned centerColIndex,
                             unsigned centerRowIndex,
                             unsigned labelIndex,
                             unsigned predictionIndex)
{

    std::ifstream file(fileName);

    std::istringstream isLine;

    if (file.is_open())
    {
        std::string row;
        std::getline(file, row, '\n'); //header
        //std::cout<<row<<std::endl;

        while (std::getline(file, row, '\n'))//read lines
        {
            std::string value;
            auto streamRow = std::istringstream(row);
            IndexT index;
            for (unsigned cIndex=1; std::getline(streamRow, value, ','); ++cIndex)
            {

                if(cIndex == imageNameIndex)
                {
                    ImageNames.emplace_back(value);
                    //std::cout<<*ImageNames.rbegin()<<" ";
                    continue;
                }
                else if (cIndex == centerColIndex)
                {

                    index[0] = std::stoi(value);

                }
                else if(cIndex == centerRowIndex)
                {
                    index[1] = std::stoi(value);
                    Centers.emplace_back(index);
                    //std::cout<<*Centers.rbegin()<<" ";

                }
                else if(cIndex == labelIndex)
                {
                    Labels.emplace_back(std::stoi(value));
                    //std::cout<<value<<" ";
                }
                else if(cIndex == predictionIndex)
                {
                    Predictions.emplace_back(std::stoi(value));
                    //std::cout<<value<<std::endl;
                }

            }

        }

    }
    file.close();


}


void ShowPrediction::SetPleuraMaskPath(const std::string& path)
{
    PleuraMasksPath = (*path.rbegin() == '/') ? path.substr(0, path.length()-1) : path;
}

void ShowPrediction::SetImagesPath(const std::string& path)
{
    ImagesPath =  (*path.rbegin() == '/') ? path.substr(0, path.length()-1) : path;
}

void ShowPrediction::SetOutputPath(const std::string& path)
{
    OutputPath =  (*path.rbegin() == '/') ? path.substr(0, path.length()-1) : path;
}

void ShowPrediction::OverlayPleuraKernel(RGBImageP& image,
                                         const GrayImageP& kernel,
                                         const IndexT& center)
{

    itk::ImageRegionConstIterator<GrayImageT> maskIt(kernel, kernel->GetLargestPossibleRegion());


    auto kernelSize = kernel->GetLargestPossibleRegion().GetSize();
    auto imageSize = image->GetLargestPossibleRegion().GetSize();


    auto lowerIndex = center;
    lowerIndex[0] -= kernelSize[0]/2;
    lowerIndex[1] -= kernelSize[1]/2;

    //validate bounds
    lowerIndex[0] = (lowerIndex[0]<0)? 0: lowerIndex[0];
    lowerIndex[1] = (lowerIndex[1]<0)? 0: lowerIndex[1];

    kernelSize[0] = (lowerIndex[0] + kernelSize[0] > imageSize[0] )? imageSize[0]-center[0] : kernelSize[0] ;
    kernelSize[1] = (lowerIndex[1] + kernelSize[1] > imageSize[1] )? imageSize[1]-center[1] : kernelSize[1];

    //std::cout<<center<<std::endl;
    //std::cout<<lowerIndex<<std::endl;


    RGBImageT::RegionType kernelRegion;
    kernelRegion.SetSize(kernelSize);
    kernelRegion.SetIndex(lowerIndex);

    itk::ImageRegionConstIterator<GrayImageT> kernelIt(kernel, kernel->GetLargestPossibleRegion());
    itk::ImageRegionIterator<RGBImageT> imageIt(image, kernelRegion);

    RGBImageT::PixelType pleuraColor;
    pleuraColor.SetRed(255);
    pleuraColor.SetGreen(0);
    pleuraColor.SetBlue(0);

    for(; !imageIt.IsAtEnd();++imageIt, ++kernelIt)
    {
        if(kernelIt.Get()==Foreground)
        {

            imageIt.Set(pleuraColor);
        }

    }

}

void ShowPrediction::SetKernelSize(unsigned size)
{
    //kernel size must be odd
    KernelSize = (size%2==0) ? size+1 : size;
}

void ShowPrediction::WritePredictions()
{

    const auto extractMaskKernel = image::ExtractNeighborhoodITK<GrayImageT>;


    std::string imageName = ImageNames[0];
    RGBImageP image = io::ReadImage<RGBImageT>(ImagesPath+"/"+ImageNames[0]);
    GrayImageP pleuraMask = io::ReadImage<GrayImageT>(PleuraMasksPath+"/"+ImageNames[0]);

    for(unsigned i=0; i < ImageNames.size(); ++i)
    {
        if(imageName != ImageNames[i])
        {

            //writing current image
            //VTKViewer::visualize<RGBImageT>(image);
            io::WriteImage<RGBImageT>(image, OutputPath+"/"+imageName);
            std::cout<<imageName<<"\n";

            //laad next image
            image = io::ReadImage<RGBImageT>(ImagesPath+"/"+ImageNames[i]);
            pleuraMask = io::ReadImage<GrayImageT>(PleuraMasksPath+"/"+ImageNames[i]);
            imageName = ImageNames[i];

        }

        if(Predictions[i]==1)
        {
            auto centerIndex = Centers[i];
            GrayImageP kernelMask;
            extractMaskKernel(pleuraMask, centerIndex, KernelSize, kernelMask);
            OverlayPleuraKernel(image, kernelMask, centerIndex);
        }

    }

    //writing the last image
    io::WriteImage<RGBImageT>(image, OutputPath+"/"+imageName);
    std::cout<<imageName<<"\n";



}







