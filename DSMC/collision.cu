#include"collision.hpp"

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

void __global__ collisionKernel(double *velocity, gridCell** cells, int cellNumber, double dt,double alpha,double cs)
{
	__shared__ float v[3*100]; 		
	__shared__ uint64_t seeds[NUM_THREADS]; 		
	__shared__ float maxRelVelocity;
	RNG rng;
	int numberParticles=cells[blockIdx.x]->number;	
	//generate a random number generator for each thread based on the cell rng
	if(threadIdx.x==0)
	{
		maxRelVelocity=0;
		for(int i=0;i<NUM_THREADS;i++)
			seeds[i]=cells[blockIdx.x]->rng.int64();
	}
	__syncthreads();
	rng=RNG(seeds[threadIdx.x]);

	//load velocities in shared memory
	for(int i=threadIdx.x;i<numberParticles;i+=blockDim.x)
	{
		int v_id=cells[blockIdx.x]->cellStartId+i;
		v[3*i+0]=velocity[3*v_id+0];
		v[3*i+1]=velocity[3*v_id+1];
		v[3*i+2]=velocity[3*v_id+2];
	}
	__syncthreads();

	//compute maximal relative velocity
	float vrMax=0;
	for(int i=threadIdx.x;i<numberParticles;i+=blockDim.x)
	{
		double3 v1=make_double3(v[3*i+0],v[3*i+1],v[3*i+2]);
		for(int j=0;j<numberParticles;j++)
		{
			double3 v2=make_double3(v[3*j+0],v[3*j+1],v[3*j+2]);
			float vr=length(v2-v1);
			vrMax=fmax(vrMax,vr);
		}
	}
	atomicMax(&maxRelVelocity,vrMax);
	__syncthreads();
	if(threadIdx.x==0)
		cells[blockIdx.x]->maxVr=maxRelVelocity;
	double3 Dr=cells[blockIdx.x]->rmax-cells[blockIdx.x]->rmin; 
	float V=Dr.x*Dr.y*Dr.z;
	for(int i=threadIdx.x;i<numberParticles;i+=blockDim.x)
	{
		for(int j=0;j<numberParticles;j++)
		{
			double3 v1=make_double3(v[3*i+0],v[3*i+1],v[3*i+2]);
			double3 v2=make_double3(v[3*j+0],v[3*j+1],v[3*j+2]);
			float vr=length(v2-v1)/2.;
			double3 vm=(v1+v2)/2.;
			if(rng.uniform()<alpha*dt/V*vr*cs/2)
			{
				double3 w=randomOnSphere(&rng);
				v1=vm+w*vr;
				v2=vm-w*vr;
			}
			__syncthreads();
			v[3*i+0]=v1.x;v[3*i+1]=v1.y;v[3*i+2]=v1.z;
			__syncthreads();
			v[3*j+0]=v2.x;v[3*j+1]=v2.y;v[3*j+2]=v2.z;
			__syncthreads();
		}
	}
	__syncthreads();
	for(int i=threadIdx.x;i<numberParticles;i+=blockDim.x)
	{
		int v_id=cells[blockIdx.x]->cellStartId+i;
		velocity[3*v_id+0]=v[3*i+0];
		velocity[3*v_id+1]=v[3*i+1];
		velocity[3*v_id+2]=v[3*i+2];
	}
}

void Grid::collision(double dt,double number, double cross_section)
{
	collisionKernel<<<numberLeafs,NUM_THREADS>>>(m_v,leafs+1,numberLeafs,dt,number/m_N,cross_section);
	gpuErrchk(cudaDeviceSynchronize());
}
