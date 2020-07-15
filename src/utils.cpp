#include"utils.h"
int cudaDeviceGetCount(void)
{
	int deviceCount=0;
	CUDA_SAFE_CALL(cuDeviceGetCount(&deviceCount));
	return deviceCount;
}

void cudaInit(void)
{
	CUDA_SAFE_CALL(cuInit(0));
}
