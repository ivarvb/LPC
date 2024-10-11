#ifndef CAST_H
#define CAST_H


namespace util
{

template<typename InputVectorT, typename OutputVectorT, unsigned size>
inline void VectorToVectorCast(const InputVectorT &input, OutputVectorT& output)
{

    for(unsigned i=0; i< size; ++i)
    {
        output[i] = input[i];
    }

}

template<typename InputVectorT, typename OutputVectorT>
inline void VectorToVectorCast(const InputVectorT &input, OutputVectorT& output, unsigned size)
{

    for(unsigned i=0; i< size; ++i)
    {
        output[i] = input[i];
    }

}

}
#endif // CAST_H
