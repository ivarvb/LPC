#include "FractalDimensionCalculator.h"

template <typename ImageT>
FractalDimensionCalculator<ImageT>::FractalDimensionCalculator()
{

}


template <typename ImageT>
void FractalDimensionCalculator<ImageT>::SetInputImage(const ImageP& InputImage)
{
    this->InputImage = nullptr;
    this->InputImage = InputImage;
}

template <typename ImageT>
void FractalDimensionCalculator<ImageT>::SetBackGround(const uint16_t& backgound)
{
    this->Background = backgound;
}

template <typename ImageT>
unsigned FractalDimensionCalculator<ImageT>::CountTiles(unsigned tileLength)
{

    unsigned tilesNumber = 0;

    const auto region = InputImage->GetRequestedRegion();

    const auto upperIndexCols = region.GetUpperIndex()[0];
    const auto upperIndexRows = region.GetUpperIndex()[1];
    const auto indexCols = region.GetIndex()[0];
    const auto indexRows = region.GetIndex()[1];
    const auto sizeRows = region.GetSize()[1];
    const auto sizeCols = region.GetSize()[0];

    typename ImageT::RegionType tile;


    const auto& getSizeRow = [tileLength, upperIndexRows, sizeRows](unsigned tileRow)
    {
        return ( tileRow + tileLength <= upperIndexRows )? tileLength : upperIndexRows - tileRow;
    };

    const auto& getSizeCol = [tileLength, upperIndexCols, sizeCols](unsigned tileCol)
    {
        return ( tileCol + tileLength <= upperIndexCols )? tileLength : upperIndexCols - tileCol;
    };


    unsigned tileSizeCol, tileSizeRow;
    for(unsigned tileRow = indexRows; tileRow <= upperIndexRows ;  tileRow += tileLength)
    {

        for(unsigned tileCol = indexCols; tileCol <= upperIndexCols ;  tileCol += tileLength)
        {

            tile.SetIndex({tileCol, tileRow});
            tileSizeCol = getSizeCol(tileCol);
            tileSizeRow = getSizeRow(tileRow);
            tile.SetSize({tileSizeCol, tileSizeRow});

            itk::ImageRegionConstIterator<ImageT> it(InputImage, tile);
            for(it; !it.IsAtEnd(); ++it)
            {
                if(it.Get() != Background)
                {
                    ++tilesNumber;
                    break;
                }

            }

        }

    }


    return tilesNumber;

}

template <typename ImageT>
double FractalDimensionCalculator<ImageT>:: GetDimension() const
{
    return Dimension;
}

template <typename ImageT>
void FractalDimensionCalculator<ImageT>::SetScale(double scale)
{

    this->ScaleFactor = scale;
}

template <typename ImageT>
void FractalDimensionCalculator<ImageT>::SetResolution(unsigned resolution)
{
    this->ScaleIterations = resolution;

}

template <typename ImageT>
void FractalDimensionCalculator<ImageT>::SetUnitTileLenght(unsigned tileLength)
{

    this->InitialTileSize = tileLength;
}


template <typename ImageT>
void FractalDimensionCalculator<ImageT>::PrintWarningsOff()
{
    PrintWarnings = false;

}

template <typename ImageT>
void FractalDimensionCalculator<ImageT>::SetScaleIterations(unsigned iterations)
{
    this->ScaleIterations = iterations;

}


template <typename ImageT>
void FractalDimensionCalculator<ImageT>::Compute()
{


    const auto& imgSize  = InputImage->GetRequestedRegion().GetSize();


    //verify if image size is larger than unit tile size
    if(imgSize[0] < InitialTileSize  || imgSize[1] < InitialTileSize )
    {
        auto tmpLength = InitialTileSize;
        //setting UnitTileLenght as the smaller side size
        InitialTileSize = (imgSize[0] > imgSize[1]) ? imgSize[1] : imgSize[0];
        if(PrintWarnings)
        {
            std::cout<<"Fractal Dimension Warning: Initial tile size reduced from "<<tmpLength<<" to "<< InitialTileSize <<std::endl;
        }

    }


    std::vector<double> scales; //tile sizes

    //discretize tile sizes by scale factor
    scales.push_back(InitialTileSize);

    for(uint16_t i=1; i <= ScaleIterations-1 ; ++i)
    {
        double s = *scales.rbegin() * ScaleFactor;
        if(s >= 1.f)
        {
            scales.push_back(std::ceil(s));
            //std::cout<<std::ceil(s)<<std::endl;
        }
        else
        {
            if(PrintWarnings)
            {
                std::cout<<"Fractal Dimension Warning: resolution adjusted from "<<ScaleIterations<<" to "<< i <<std::endl;
            }
            break;

        }

    }


    //lambda
    //Fractal dimension function: tiles = number of tiles different from background
    const auto& fractalDim = [](unsigned largeTiles, double largeScale, unsigned smallTiles, double smallScale) //double because of implicit conversion float to double
    {
        return ( std::log(smallTiles) - std::log(largeTiles)) / (std::log(1/smallScale) - std::log(1/largeScale));
    };


    //computing fractal dimension for each scale
    std::vector<double> dimensions;
    double currentScale = 1.f;
    for(auto it = scales.begin()+1; it != scales.end(); ++it)
    {
        auto largeTiles = CountTiles(*(it-1)); //large tiles
        auto smallTiles = CountTiles(*(it)); //small tiles

        if(largeTiles == 0 || smallTiles == 0) //
        {
            //std::cout<<"Uniform here"<<std::endl;
            dimensions.push_back(0.f);
        }
        else
        {
            dimensions.push_back(fractalDim(largeTiles, currentScale, smallTiles, currentScale*ScaleFactor));
        }
        currentScale*=ScaleFactor;

    }

    //average fractal dimension
    Dimension = std::accumulate(dimensions.begin(), dimensions.end(), 0.f) / dimensions.size();


    //std::cout<<Dimension<<std::endl;


}


