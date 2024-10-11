#ifndef FRACTALDIMENSIONFILTER_H
#define FRACTALDIMENSIONFILTER_H

#include <itkImage.h>
#include <cmath>
#include <numeric>
#include <set>
#include <itkImageRegionConstIterator.h>


//#include "Math.h"


template<typename ImageT>
class FractalDimensionCalculator
{

    using ImageP = typename ImageT::Pointer;




public:
    FractalDimensionCalculator();
    void SetInputImage(const ImageP& InputImage);

    void SetUnitTileLenght(unsigned lenght); //in pixels
    void SetResolution(unsigned reolution);
    void SetScale(double scale);
    void PrintWarningsOff();
    void SetBackGround(const uint16_t& backgound);
    void SetScaleIterations(unsigned iterations);

    double GetDimension() const;

    void Compute();




private:

    ImageP InputImage;

    unsigned InitialTileSize{10}; //Initial tile size
    unsigned ScaleIterations{3}; //number of tile reduction (scaling)
    double   ScaleFactor{0.5}; //must be in [0 1];
    double   Dimension{0}; //result
    bool     PrintWarnings{true};
    uint16_t Background{0};

    unsigned CountTiles(unsigned unitTileLenght);


};

template class FractalDimensionCalculator<itk::Image<unsigned char, 2>>;
template class FractalDimensionCalculator<itk::Image<unsigned, 2>>;

#endif // FRACTALDIMENSIONFILTER_H
