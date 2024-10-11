#ifndef TESTER_H
#define TESTER_H

#include <string>
#include <iostream>

#include <dlib/array2d.h>
#include <dlib/svm.h>

#include "../util/InputOutput.h"

#include <itkImage.h>

class Tester
{

    using GrayImageT = itk::Image<unsigned, 2>;
    using IndexT = GrayImageT::IndexType;
    using CentersT  = std::vector<IndexT>;

    using FeaturesT  = dlib::matrix<double>;
    using KernelT = dlib::radial_basis_kernel<FeaturesT>;
    using DecisionFunctionT = dlib::decision_function<KernelT>;
    using FunctionT = dlib::normalized_function<DecisionFunctionT>;

    using FeaturesVectorT = std::vector<FeaturesT>;

public:
    Tester();
    void ReadLearnedFunction(const std::string& fileName);
    void ReadFeaturesCSV(const std::string& fileName,
                         unsigned imageColumn,
                         unsigned centerBegin,
                         unsigned featuresBegin,
                         unsigned featuresEnd,
                         unsigned labelColumn);

    void Process();


private:
    FunctionT LearnedFunction;
    FeaturesVectorT  FeaturesVector;
    std::vector<double> LabelsGroundTruth;
    std::vector<double> Labels;
    std::vector<std::string> ImagesNames;
    CentersT   CentersVector;





};

#endif // TESTER_H
