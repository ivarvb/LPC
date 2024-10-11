#ifndef TRAINER_H
#define TRAINER_H

#include <string>
#include <iostream>

#include <dlib/array2d.h>
#include <dlib/svm.h>

#include "../util/InputOutput.h"

class Trainer
{

    using SampleT  = dlib::matrix<double>;
    using SamplesT = std::vector<SampleT>;
    using KernelT = dlib::radial_basis_kernel<SampleT>;
    using DecisionFunctionT = dlib::decision_function<KernelT>;
    using FunctionT = dlib::normalized_function<DecisionFunctionT>;


public:
    Trainer();

    void ReadFeaturesCSV(const std::string& fileName,
                         unsigned featuresBeginIndex,
                         unsigned featuresEndIndex,
                         unsigned labelColumn);
    void ProcessKrr();
    void ProcessSVMRadial();
    void ProcessSVM();
    void WriteLearnedFunction(const std::string& fileName);


private:

     SamplesT  Samples;
     FunctionT LearnedFunction;
     std::vector<double> Labels;


};

#endif // TRAINER_H
