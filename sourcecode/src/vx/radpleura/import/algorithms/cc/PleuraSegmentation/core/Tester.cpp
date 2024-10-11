#include "Tester.h"

Tester::Tester()
{

}

void Tester::ReadLearnedFunction(const std::string& fileName)
{
    dlib::deserialize(fileName)>>LearnedFunction;
}

void Tester::ReadFeaturesCSV(const std::string& fileName,
                             unsigned imageColumn,
                             unsigned centerBegin,
                             unsigned featuresBegin,
                             unsigned featuresEnd,
                             unsigned labelColumn)
{

    std::ifstream file(fileName);

    unsigned featuresSize = featuresEnd - featuresBegin+1;
    std::cout<<featuresSize<<std::endl;
    FeaturesT features(featuresSize,1);
    std::string line;
    std::string value;
    std::istringstream iLine;
    IndexT center;
    if (file.is_open())
    {
        while (std::getline(file, line))
        {
            iLine = std::istringstream(line);

            for (unsigned lIndex=1; std::getline(iLine, value, ','); ++lIndex)
            {

                if(lIndex == imageColumn)
                {
                    ImagesNames.push_back(value);

                }

                else if(lIndex == labelColumn)
                {
                    LabelsGroundTruth.push_back(std::stod(value));
                    //std::cout<<*Labels.rbegin()<<std::endl;
                }
                else if (lIndex == centerBegin)
                {
                    center[0] = std::stoi(value);
                    std::getline(iLine, value, ',');
                    center[0] = std::stoi(value);
                    ++lIndex;
                    CentersVector.push_back(center);
                }

                else if (lIndex == featuresBegin)
                {
                    features(0) = static_cast<double>(std::stold(value)); //verify this
                    for (unsigned fIndex=1; fIndex < featuresSize ; ++fIndex)
                    {
                        std::getline(iLine, value, ',');
                        features(fIndex) = std::stod(value);
                        ++lIndex;
                    }
                    FeaturesVector.push_back(features);
                    //std::cout<<*FeaturesVector.rbegin()<<std::endl;

                }
                else
                {
                    continue;
                }

            }

        }
    }

    file.close();
}



void Tester::Process()
{


    Labels.clear();
    Labels.reserve(FeaturesVector.size());

    for(const auto & features: FeaturesVector )
    {
         Labels.push_back( LearnedFunction(features) );
         std::cout<<*Labels.rbegin()<<std::endl;
    }


}



















