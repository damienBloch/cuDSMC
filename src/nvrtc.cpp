#include"nvrtc.hpp"

CUmodule loadModule(char *prgStr)
{
	CUmodule module;

	nvrtcProgram prg;
	NVRTC_SAFE_CALL(nvrtcCreateProgram(&prg,prgStr,NULL,0,NULL,NULL)); 

	const char *compilOpts[] = {"--gpu-architecture=compute_50","--fmad=false"};
	auto err=nvrtcCompileProgram(prg,2,compilOpts); 
	if(err!=NVRTC_SUCCESS)
	{
		std::string errorTypeStr(nvrtcGetErrorString(err));
		size_t logSize;
		nvrtcGetProgramLogSize(prg, &logSize);
		char *log = new char[logSize];
		NVRTC_SAFE_CALL(nvrtcGetProgramLog(prg, log));
		std::string logString(log);
		throw std::runtime_error(errorTypeStr+std::string("\n")+std::string(prgStr)+std::string("\n")+logString);
		delete log;
	} 
	else
	{

		size_t ptxSize;
		NVRTC_SAFE_CALL(nvrtcGetPTXSize(prg, &ptxSize));
		char *ptx = new char[ptxSize];
		NVRTC_SAFE_CALL(nvrtcGetPTX(prg, ptx));
		NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prg));

		CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
		delete ptx;
	}
	return module;
}
