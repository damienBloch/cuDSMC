#include"histogram.h"

void __global__ setKeys(double *r, unsigned int* keys,int N,double min_x,double max_x, int NX,double min_y,double max_y, int NY,double min_z,double max_z, int NZ)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<N)
	{
		int idx=NX*((r[3*id+0]-min_x)/(max_x-min_x));
		int idy=NY*((r[3*id+1]-min_y)/(max_y-min_y));
		int idz=NZ*((r[3*id+2]-min_z)/(max_z-min_z));
		if(!isfinite(min_x)) idx=0;
		if(!isfinite(min_y)) idy=0;
		if(!isfinite(min_z)) idz=0;
		if(idx<0 || idy<0 || idz<0 || idx>=NX || idy >=NY || idz>=NZ)
			keys[id]=NX*NY*NZ;
		else
			keys[id]=idx+NX*(idy+NY*idz);
	}
}

void __global__ setID(unsigned int* cells, int N)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<N)
	{
		cells[id]=id;
	}
}

Histogram::Histogram(double min_x,double max_x, int NX,double min_y,double max_y, int NY,double min_z,double max_z, int NZ):
min_x(min_x),max_x(max_x),NX(NX),min_y(min_y),max_y(max_y),NY(NY),min_z(min_z),max_z(max_z),NZ(NZ)
{
	m_hist=NULL;
	gpuErrchk(cudaMallocManaged(&m_hist,NX*NY*NZ*sizeof(unsigned int)));
}

Histogram::~Histogram()
{
	if(m_hist!=NULL)
		gpuErrchk(cudaFree(m_hist));
}

unsigned int* Histogram::bin(double* r, int N)
{
	unsigned int *keys=NULL;
	gpuErrchk(cudaMallocManaged(&keys,N*sizeof(int)));
	
	setKeys<<<N,1>>>(r,keys,N,min_x,max_x,NX,min_y,max_y,NY,min_z,max_z,NZ);
	thrust::device_ptr<unsigned int> d_keys(keys);
	thrust::device_ptr<unsigned int> d_hist(m_hist);
	thrust::sort(thrust::device,d_keys,d_keys+N);
	setID<<<NX*NY*NZ,1>>>(m_hist,NX*NY*NZ);
	thrust::upper_bound(thrust::device,d_keys,d_keys+N,d_hist,d_hist+NX*NY*NZ,d_hist);
	thrust::adjacent_difference(thrust::device,d_hist,d_hist+NX*NY*NZ,d_hist);

	cudaDeviceSynchronize();

	if(keys!=NULL)
		gpuErrchk(cudaFree(keys));
	cudaDeviceSynchronize();
	return m_hist;
}
