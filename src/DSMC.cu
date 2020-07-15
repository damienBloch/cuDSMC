#include"DSMC.hpp"

void DSMC::initCuda(int device)
{
	if(device>=cudaDeviceGetCount())throw std::runtime_error("Device number "+std::to_string(device)+" exceeds the number of available devices "+std::to_string(cudaDeviceGetCount()));

	CUdevice cuDevice;
	CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, device));

	CUDA_SAFE_CALL(cuCtxCreate(&mCuContext, 0, cuDevice));
	cudaSetDevice(device);
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth,gridDepth);
}

DSMC::DSMC(py::array_t<double>& r,py::array_t<double>& v,double t0,int device)
{
	initCuda(device);

	//check if position & speed dimensions are ok 
	py::buffer_info buf_r = r.request(), buf_v = v.request();
	if(buf_r.ndim !=1 || buf_r.ndim !=1)
		throw std::runtime_error("Number of dimensions must be one");
	if(buf_r.size !=buf_v.size)
		throw std::runtime_error("Input shape must match");
	if(buf_r.size%3!=0)
		throw std::runtime_error("Input must be of dimension (3,N)");
	m_NT=buf_r.size/3;

	//allocate & copy positions & speed to device
	mPositionArray=thrust::device_vector<double>(3*m_NT);
	gpuErrchk(cudaMemcpy(thrust::raw_pointer_cast(&mPositionArray[0]),buf_r.ptr,sizeof(double)*3*m_NT,cudaMemcpyHostToDevice));

	mVelocityArray=thrust::device_vector<double>(3*m_NT);
	gpuErrchk(cudaMemcpy(thrust::raw_pointer_cast(&mVelocityArray[0]),buf_v.ptr,sizeof(double)*3*m_NT,cudaMemcpyHostToDevice));
	//thrust::device_vector<double> test(5);
	//gpuErrchk(cudaMemcpy(m_r,buf_r.ptr,sizeof(double)*3*m_NT,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&m_r,m_NT*3*sizeof(double)));
	gpuErrchk(cudaMalloc(&m_v,m_NT*3*sizeof(double)));
	gpuErrchk(cudaMemcpy(m_r,buf_r.ptr,sizeof(double)*3*m_NT,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(m_v,buf_v.ptr,sizeof(double)*3*m_NT,cudaMemcpyHostToDevice));

	m_t=t0;
	//creat adaptative grid for particle
	double3 rmin=make_double3(-500e-6,-500e-6,-500e-6);
	double3 rmax=make_double3(500e-6,500e-6,500e-6);
	m_grid=new Grid(m_r,m_v,m_NT,rmin,rmax);	
}

py::array_t<double> DSMC::getPositions()
{
	auto ret = py::array_t<double>(3*m_NT);
	gpuErrchk(cudaMemcpy(ret.request().ptr,PTR(mPositionArray),sizeof(double)*3*m_NT,cudaMemcpyDeviceToHost));
	return ret;
}

py::array_t<double> DSMC::getSpeeds()
{
	auto ret = py::array_t<double>(3*m_NT);
	gpuErrchk(cudaMemcpy(ret.request().ptr,PTR(mVelocityArray),sizeof(double)*3*m_NT,cudaMemcpyDeviceToHost));
	return ret;
}

DSMC::~DSMC()
{
	if(m_advection.moduleLoaded)
		CUDA_SAFE_CALL(cuModuleUnload(m_advection.module));
	if(m_grid!=NULL)
		delete m_grid;
	CUDA_SAFE_CALL(cuCtxDestroy(mCuContext));
}



void DSMC::loadPotential(char *str)
{
	if(m_advection.moduleLoaded)
		CUDA_SAFE_CALL(cuModuleUnload(m_advection.module));
	m_advection.module=loadModule(str);
	CUDA_SAFE_CALL(cuModuleGetFunction(&(m_advection.kernel), m_advection.module,"advectionKernel"));
	CUDA_SAFE_CALL(cuModuleGetFunction(&(m_energyKernel), m_advection.module,"computeEnergy"));
	m_advection.moduleLoaded=true;
	CUDA_SAFE_CALL(cuOccupancyMaxPotentialBlockSize( &(m_advection.minGridSize), &(m_advection.blockSize),m_advection.kernel,NULL, 0, 0));
}

py::array_t<unsigned int> DSMC::makeHistogram(double min_x,double max_x, int NX,double min_y,double max_y, int NY,double min_z,double max_z, int NZ)
{
	Histogram hist(min_x,max_x,NX,min_y,max_y,NY,min_z,max_z,NZ);
	thrust::device_vector<unsigned int>& h=hist.bin(mPositionArray);
	auto result = py::array_t<unsigned int>(NX*NY*NZ);
	py::buffer_info buf_result = result.request();
	gpuErrchk(cudaMemcpy(buf_result.ptr,PTR(h),sizeof(unsigned int)*NX*NY*NZ,cudaMemcpyDeviceToHost));
	return result;
}

double DSMC::update(double t)
{
	if(!m_advection.moduleLoaded)
		throw std::runtime_error("Potential was not correctly set");
	else
	{
		double dt=1e-4;
		while(m_t<t)
		{
			double *ptrPosition=thrust::raw_pointer_cast(&mPositionArray[0]);
			double *ptrVelocity=thrust::raw_pointer_cast(&mVelocityArray[0]);
			void *args[] = { &ptrPosition, &ptrVelocity, &m_NT, &dt, &m_t };
			CUDA_SAFE_CALL(cuLaunchKernel(m_advection.kernel,
						(m_NT+m_advection.blockSize-1)/m_advection.blockSize, 1, 1, 
						m_advection.blockSize, 1, 1, 
						0, NULL, 
						args,
						0));
/*			m_grid->updateGrid();
			m_grid->collision(dt,m_NP,m_cs);*/
			m_t+=dt;	
		}
	}
	return m_t;
}

py::array_t<gridCell> DSMC::getGrid()
{
	gridCell *cells=m_grid->getCells();	
	auto result = py::array_t<gridCell>(m_grid->numberLeafs);
	gpuErrchk(cudaMemcpy(result.request().ptr,cells,sizeof(gridCell)*m_grid->numberLeafs,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(cells));
	return result;
}
