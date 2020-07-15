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
	mHist=thrust::device_vector<unsigned int>(NX*NY*NZ);
}

thrust::device_vector<unsigned int>& Histogram::bin(thrust::device_vector<double> r)
{
	unsigned int N=r.size()/3;
	thrust::device_vector<unsigned int> keys(N);
	
	setKeys<<<N,1>>>(PTR(r),PTR(keys),N,min_x,max_x,NX,min_y,max_y,NY,min_z,max_z,NZ);
	thrust::sort(thrust::device,keys.begin(),keys.end());
	setID<<<NX*NY*NZ,1>>>(PTR(mHist),NX*NY*NZ);
	thrust::upper_bound(thrust::device,keys.begin(),keys.end(),mHist.begin(),mHist.end(),mHist.begin());
	thrust::adjacent_difference(thrust::device,mHist.begin(),mHist.end(),mHist.begin());
	cudaDeviceSynchronize();
	return mHist;
}
