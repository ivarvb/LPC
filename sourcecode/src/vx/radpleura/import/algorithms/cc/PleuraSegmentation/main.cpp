#include <iostream>
#include <memory>
#include <cinttypes>
#include <cstring>

//local includes
#include "core/PreProcessor.h"
#include "core/BoundariesExtractor.h"
#include "core/FeatureExtractor.h"
#include "core/Trainer.h"
#include "core/Tester.h"
#include "core/ShowPrediction.h"

#include "core/Pipe.h"

/*



*/


int main(int argc, char* *argv) {
    std::string path(argv[1]);
    unsigned erodeRadius = atoi(argv[2]);
    //double tilepercentage = atof(argv[3]);

    /* std::string dataset = "dataset_1"; */
    //std::string pathdir = "/mnt/sda6/software/frameworks/data/lha/dataset_4/";

/* 
    //input folder images
    //output folder required "images_cleaned"
    PreProcessor preProcessor;
    preProcessor.SetLThreshold(98);
    preProcessor.SetBThreshold(-1);
    preProcessor.SetAThreshold(1);
    preProcessor.SetInputDatasetPath(pathdir+"images/");
    preProcessor.SetOutputDatasetPath(pathdir+"images_cleaned/");
    preProcessor.Process();
 */


/*  
    //input folder image_cleaned
    //output folders required "boundaries" and "masks"
    BoundariesExtractor boundaryExtractor;
    boundaryExtractor.SetInputDatasetPath(pathdir+"images_cleaned/");
    boundaryExtractor.SetOutputDatasetPath(pathdir+"boundaries/");
    boundaryExtractor.SetOutputMaskPath(pathdir+"masks/");
    boundaryExtractor.SetGaussSigma(0.5);
    boundaryExtractor.SetSmallComponentsThreshold(500);
    boundaryExtractor.SetThinBoundariesOff();
    boundaryExtractor.Process();
 */


/* 

    //input folders boundaries
    //images_cleaned
    //labels
    //masks
    //output folder boundary_masks/<erode radius size>
    //csv
    unsigned kernelSize = 50;
    unsigned erodeRadius = 30;
    //// std::string setType = "test"; 
    FeatureExtractor featureExtractor;
    featureExtractor.SetBoundariesPath(pathdir+"boundaries/");
    featureExtractor.SetImagesPath(pathdir+"images_cleaned/");
    featureExtractor.SetLabelsPath(pathdir+"labels/"); //use exiftool -all *.tiff to remove all metadata...
    featureExtractor.SetMasksPath(pathdir+"masks/");
    featureExtractor.SetPleuraMasksPath(pathdir+"boundary_masks/erode_radius_"+std::to_string(erodeRadius)+"/");
    featureExtractor.SetKernelSize(kernelSize);
    featureExtractor.SetErodeRadius(erodeRadius);
    featureExtractor.Process();
    //featureExtractor.WriteFeaturesCSV(pathdir+"/csv/kernel_size_"+std::to_string(kernelSize)+".csv", true);
*/





/*

    //input folders images,
    unsigned kernelSize = 250;
    ShowPrediction showPredictions;
    showPredictions.SetPleuraMaskPath("/home/oscar/data/biopsy/tiff/test/pleura_masks_20/");
    showPredictions.SetImagesPath("/home/oscar/data/biopsy/tiff/test/images");
    showPredictions.SetOutputPath("/home/oscar/data/biopsy/tiff/test/classifications_"+std::to_string(kernelSize));
    showPredictions.ReadCSV("/home/oscar/data/biopsy/tiff/test/csv/test_"+std::to_string(kernelSize)+"_classification.csv", 2, 3,4, 71, 72);
    showPredictions.SetKernelSize(kernelSize);
    showPredictions.WritePredictions();
*/





/*

    Trainer trainer;
    trainer.ReadFeaturesCSV("/home/oscar/data/biopsy/tiff/test/csv/fractal_lbp_50.csv", 4, 63, 64);
    trainer.ProcessSVMRadial();
    //trainer.ProcessSVM();
    //trainer.WriteLearnedFunction("/home/oscar/data/biopsy/tiff/test/learned_function.dat");
*/

    /*
    auto tester = std::make_unique<Tester>();
    tester->ReadFeaturesCSV("/home/oscar/data/biopsy/tiff/test/tranning_set.csv", 1,2,3,6,7);
    tester->ReadLearnedFunction("/home/oscar/data/biopsy/tiff/test/learned_function.dat");
    */


    /* "/mnt/sda6/software/frameworks/data/lha/dataset_4/testxx" */
    std::cout<<"BEGIN"<<std::endl;

    Pipe pip;
    pip.setPath(path);
    pip.setErodeRadius(erodeRadius);
    pip.execute();

    std::cout<<"END"<<std::endl;

    return 0;
}
