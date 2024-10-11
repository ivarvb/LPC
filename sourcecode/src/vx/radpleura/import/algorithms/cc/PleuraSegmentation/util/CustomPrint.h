#ifndef CUSTOMPRINT_H
#define CUSTOMPRINT_H


#include <iostream>
#include <iomanip>

namespace io
{


inline void printProgress(const std::string& message, const unsigned& progress,  short size = 40)
{

    short diff = size - static_cast<short>(message.size()+1);
    std::cout << message<<": "<< std::setw(diff) << progress <<"%"<<std::flush<<std::endl;
}

inline void printLn(const std::string& message)
{
    std::cout<<message<<std::endl;
}

inline void printWait(std::string message, std::string timeMessage = "a while")
{

    std::cout << message + ": please wait, this may take "+timeMessage<<std::flush;
    std::cout<<std::endl;

}

inline void printInfo(std::string subject, std::string message)
{

    std::cout << subject + ": "+message<<std::flush;
    std::cout<<std::endl;

}


inline void printOK(std::string message, short size = 50)
{

    short diff = size - static_cast<short>(message.size());
    std::cout << message << std::setw(diff) << "[OK]" << std::endl;

}


inline void printFail(std::string message, std::string extraInformation="" , short size = 40)
{

    short diff = size - static_cast<short>(message.size());
    std::cerr << message << std::setw(diff) << "[Fail]"<<std::endl<<std::flush;
    if(extraInformation.size() > 0)
    {
        std::cerr <<": "<< extraInformation<<std::endl;
    }


}


template <typename VectorT, unsigned size>
inline void PrintVector(const VectorT& vector)
{



   for(unsigned i=0;i< size -1;++i)
   {
       std::cout<<vector[i]<<' ';
   }
   std::cout<<vector[size-1]<<'\n';

}
template <typename VectorT>
inline void PrintVector(const VectorT& vector, unsigned size)
{

   for(unsigned i=0; i< size -1 ;++i)
   {
       std::cout<<vector[i]<<" ";
   }
   std::cout<<vector[size-1]<<"\n";

}



};

#endif // CUSTOMPRINT_H
