#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

int main(int argc, char * argv[]){
  if (argc != 3){
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " <InputImageFile> <OutputImageFile> <sigma>" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 2;

  const char * inputFileName = argv[1];
  const char * outputFileName = argv[2];
  const float  sigmaValue = 1.0;

  using PixelType = unsigned char;
  using ImageType = itk::Image<PixelType, Dimension>;

  using ReaderType = itk::ImageFileReader<ImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFileName);

  using FilterType = itk::SmoothingRecursiveGaussianImageFilter<ImageType, ImageType>;
  FilterType::Pointer smoothFilter = FilterType::New();

  smoothFilter->SetSigma(sigmaValue);
  smoothFilter->SetInput(reader->GetOutput());

  using WriterType = itk::ImageFileWriter<ImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(smoothFilter->GetOutput());
  writer->SetFileName(outputFileName);

  try{
    writer->Update();
  }
  catch (itk::ExceptionObject & error){
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
