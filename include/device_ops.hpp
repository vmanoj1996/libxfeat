#include <cuda_runtime.h>

struct BatchNormRelu
{
public:
    float bias;
    float scale;

   __device__ float forward(float u)
   {
       float y = scale * u + bias;
       return (y > 0) ? y : 0.0f;
   }

};

struct Identity
{
public:

   __device__ float forward(float u)
   {
       return u;
   }

};
