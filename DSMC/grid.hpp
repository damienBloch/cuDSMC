#ifndef GRID_H
#define GRID_H
#include"utils.h"
#include"allocator.hpp"

#include<thrust/sort.h>
#include<thrust/unique.h>
#include<thrust/binary_search.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/device_vector.h>
#include<thrust/execution_policy.h>
#include<thrust/remove.h>
#define gridDepth (10)
#ifdef __CUDACC__
# define ALIGN(x) __align__(x)
#else
# define ALIGN(x) alignas(x)
#endif

struct RNG{
	uint64_t u,v,w;
	__host__ __device__ RNG(){};
	__host__ __device__ RNG(uint64_t j):v(4101842887655102017LL), w(1) {
		u = j ^ v; int64();
		v = u; int64();
		w = v; int64();
	}
	__host__ __device__ inline uint64_t int64() {
		u = u * 2862933555777941757LL + 7046029254386353087LL;
		v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
		w = 4294957665U*(w & 0xffffffff) + (w >> 32);
		uint64_t x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
		return (x + v) ^ w;
	}
	__host__ __device__ inline double uniform() { return 5.42101086242752217E-20 * int64(); }
	__host__ __device__ inline uint32_t int32() { return (uint32_t)int64(); }
};

__device__ inline double3 randomOnSphere(RNG *rng)
{
	double2 u;
	do{
		u.x=2*rng->uniform()-1;	
		u.y=2*rng->uniform()-1;	
	}while(u.x*u.x+u.y*u.y>1-1e-8);
	double3 r=make_double3(2*u.x*sqrt(1-u.x*u.x-u.y*u.y),2*u.y*sqrt(1-u.x*u.x-u.y*u.y),1-2*(u.x*u.x+u.y*u.y));
	return r;
};


struct gridCell{
		int number;
		gridCell* parent;
		gridCell* children[8];
		int level;
		double3 rmin,rmax;
		unsigned int keyMin,keyMax;
		unsigned int* cellStart,*cellEnd;
		unsigned int cellStartId,cellEndId;
		int id;
		RNG rng;
		double maxVr;
};

struct is_null
{
  __host__ __device__
  bool operator()(const void* x)
  {
    return x == NULL;
  }
};

class Grid{
	public:
		Grid(){};
		Grid(double* r,double *v,unsigned int N,double3 rmin,double3 rmax);
		void sortParticles();
		~Grid();
		gridCell* getCells();
		unsigned int numberLeafs;
		void updateGrid();
		void collision(double dt,double number, double cross_section);
	private:
		unsigned int *m_zoId;
		double *m_r,*m_v;
		unsigned int m_N;
		double3 rmin,rmax;
		gridCell** leafs;
		gridCell** root;
		cached_allocator alloc;
		void compactLeafs(void);
};

struct Segment{
	double r[3];
};

#endif
