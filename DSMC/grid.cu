#include"grid.hpp"

unsigned int __device__ mixBits(unsigned int id1,unsigned int id2,unsigned int id3,int depth)
{
	/*mix bits of id1,id2,id3 in order to build Z-ordering curve
	ex:
	id1 =      1  0  1
	id2 =     0  1  1
	id3 =    0  0  1
	result = 001010111 
	*/
	unsigned int result=0;
	for(int i=0;i<depth;i++)
	{
		result|=((id1&(1<<i))>>i)<<(3*i+0);
		result|=((id2&(1<<i))>>i)<<(3*i+1);
		result|=((id3&(1<<i))>>i)<<(3*i+2);
	}
	return result;
}

unsigned int __device__ getZOId(double3 r,double3 rmin, double3 rmax)
{
	unsigned int key;
	int idx=(1<<gridDepth)*((r.x-rmin.x)/(rmax.x-rmin.x));
	int idy=(1<<gridDepth)*((r.y-rmin.y)/(rmax.y-rmin.y));
	int idz=(1<<gridDepth)*((r.z-rmin.z)/(rmax.z-rmin.z));
	if(idx<0 || idy<0 || idz<0 || idx>=(1<<gridDepth) || idy >=(1<<gridDepth) || idz>=(1<<gridDepth))
			key=(unsigned int)(1<<(3*gridDepth));
	else
			key=mixBits(idx,idy,idz,gridDepth);
	return key;
}

void __global__ setZoId(double *r, unsigned int N,unsigned int *keys, double3 rmin,double3 rmax)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<N)
	{
		keys[id]=getZOId(make_double3(r[3*id+0],r[3*id+1],r[3*id+2]),rmin,rmax);
	}
}

void __global__ destroyTree(gridCell** root)
{
	int id=4*threadIdx.z+2*threadIdx.y+threadIdx.x;
	if(((root[id])->children[0])!=NULL)
		destroyTree<<<1,dim3(2,2,2)>>>(root[id]->children);
	delete root[id];
}

gridCell* __device__ createCell(gridCell *parent,gridCell** parentChildren,int id,int level,int seed,double3 rminParentCell,double3 rmaxParentCell,int divider,unsigned int parentMin,unsigned int parentMax,unsigned int* keys, unsigned int N)
{
	gridCell* currentCell=new gridCell;	
	currentCell->parent=parent;
	currentCell->id=id;
	currentCell->rng=RNG(seed+id);
	parentChildren[id]=currentCell;
	for(int i=0;i<8;i++)currentCell->children[i]=NULL;
	currentCell->level=level;
	currentCell->rmin=rminParentCell+make_double3(id%2,(id/2)%2,id/4)*(rmaxParentCell-rminParentCell)/make_double3(divider,divider,divider);
	currentCell->rmax=currentCell->rmin+(rmaxParentCell-rminParentCell)/make_double3(divider,divider,divider);
	unsigned int keyMin=parentMin;
	unsigned int keyMax=parentMax;
	if(level>0)
	{
		keyMin&=~(7<<(3*(gridDepth-level)));
		keyMin|=id<<(3*(gridDepth-level));

		keyMax&=~(7<<(3*(gridDepth-level)));
		keyMax|=id<<(3*(gridDepth-level));
	}
	thrust::device_ptr<unsigned int> d_keys(keys);
	unsigned int *cellStart=thrust::raw_pointer_cast(thrust::lower_bound(thrust::seq,d_keys,d_keys+N,keyMin));
	unsigned int *cellEnd=thrust::raw_pointer_cast(thrust::upper_bound(thrust::seq,d_keys,d_keys+N,keyMax));
	unsigned int cellParticleNumber=cellEnd-cellStart;
	currentCell->number=cellParticleNumber;
	currentCell->keyMin=keyMin;currentCell->keyMax=keyMax;
	currentCell->cellStart=cellStart;currentCell->cellEnd=cellEnd;
	return currentCell;

}

void __global__ buildTree(unsigned int* keys, unsigned int N,double3 rminParentCell, double3 rmaxParentCell,unsigned int parentMin, unsigned int parentMax,unsigned int level,gridCell* parent,int maxParticles,gridCell* leafs[],unsigned long long seed,gridCell** root,unsigned int maxN)
{
	gridCell* currentCell;
	if(level>0)
		currentCell=createCell(parent,parent->children,4*threadIdx.z+2*threadIdx.y+threadIdx.x,level,seed,rminParentCell,rmaxParentCell,2,parentMin,parentMax,keys,N);
	else
		currentCell=createCell(NULL,root,0,level,seed,rminParentCell,rmaxParentCell,1,parentMin,parentMax,keys,N);
	if(currentCell->number>maxParticles)
	{
		buildTree<<<1,dim3(2,2,2)>>>(currentCell->cellStart,currentCell->number,currentCell->rmin,currentCell->rmax,currentCell->keyMin,currentCell->keyMax,currentCell->level+1,currentCell,maxParticles,leafs,currentCell->rng.int64(),root,maxN);
	}
	else
	{
		//find a free place to store a pointer to the current cell in a buffer
		typedef unsigned long long int uint64;	
		uint64 adr;
		gridCell** pos;
		do{
			pos=leafs+(currentCell->rng.int64()%maxN);
			adr=atomicCAS((uint64*)pos,(uint64)NULL,(uint64)currentCell);
		}while(adr!=NULL);
	}
}

Grid::Grid(double *r,double *v, unsigned int N,double3 rmin, double3 rmax):m_r(r),m_v(v),m_N(N),rmin(rmin),rmax(rmax)
{
	m_zoId=NULL;
	gpuErrchk(cudaMalloc(&m_zoId,N*sizeof(unsigned int)));	
	leafs=NULL;
	gpuErrchk(cudaMallocManaged(&leafs,N*sizeof(gridCell*)));	
	root=NULL;
	gpuErrchk(cudaMalloc(&root,sizeof(gridCell*)));	
	sortParticles();
	gpuErrchk(cudaMemset(leafs,0,N*sizeof(gridCell*)));
	buildTree<<<1,1>>>(m_zoId,m_N,rmin,rmax,0,(1<<(3*gridDepth))-1,0,NULL,100,leafs,0,root,m_N);
	compactLeafs();
	gpuErrchk(cudaDeviceSynchronize());
}

void __global__ gatherCells(gridCell* cells,unsigned int N,gridCell** leafs)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<N)
		cells[id]=*(leafs[id]);
}



Grid::~Grid()
{
	if(m_zoId!=NULL)
		gpuErrchk(cudaFree(m_zoId));
	if(leafs!=NULL)
		gpuErrchk(cudaFree(leafs));
	if(root!=NULL)
	{
		destroyTree<<<1,1>>>(root);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaFree(root));
	}
}

void __global__ splitGrid(unsigned int* keys, unsigned int NParts,gridCell** leafs,unsigned int NLeafs,unsigned int maxParticles)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<NLeafs)
	{

		gridCell* stack[73]={leafs[id]};
		int counter=1;
		int s=true;
		while(counter>0)
		{
			gridCell* currentCell = stack[counter-1];counter--;
			thrust::device_ptr<unsigned int> d_keys(keys);
			currentCell->cellStart=thrust::raw_pointer_cast(thrust::lower_bound(thrust::seq,d_keys,d_keys+NParts,currentCell->keyMin));
			currentCell->cellEnd=thrust::raw_pointer_cast(thrust::upper_bound(thrust::seq,d_keys,d_keys+NParts,currentCell->keyMax));
			currentCell->number=currentCell->cellEnd-currentCell->cellStart;
			if(currentCell->number>maxParticles)
			{
				if(s)
				{
					leafs[id]=NULL;
					s=false;
				}
				for(int i=0;i<8;i++)
				{
					gridCell* child=createCell(currentCell,currentCell->children,i,currentCell->level+1,currentCell->rng.int64(),currentCell->rmin,currentCell->rmax,2,currentCell->keyMin,currentCell->keyMax,currentCell->cellStart,currentCell->number);
					if(child->number>maxParticles && counter<73)
					{
						stack[counter]=child;
						counter++;
					}
					else
					{
						typedef unsigned long long int uint64;	
						uint64 adr;
						gridCell** pos;
						do{
							pos=leafs+(currentCell->rng.int64()%(NParts-1));
							adr=atomicCAS((uint64*)pos,(uint64)NULL,(uint64)child);
						}while(adr!=NULL);
					}
				}
			}
		}
	}
}

void __global__ removeCells(gridCell** leafs, unsigned int NLeafs)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<NLeafs)
	{
		if(leafs[id]->id<0)
		{
			delete leafs[id]; 
			leafs[id]=NULL;
		}
	}
}

void __global__ calculateNumber(unsigned int* keys, unsigned int NParts,gridCell** leafs,unsigned int NLeafs,unsigned int maxParticles)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<NLeafs)
	{
		gridCell* currentCell=leafs[id];	
		thrust::device_ptr<unsigned int> d_keys(keys);
		currentCell->cellStart=thrust::raw_pointer_cast(thrust::lower_bound(thrust::seq,d_keys,d_keys+NParts,currentCell->keyMin));
		currentCell->cellEnd=thrust::raw_pointer_cast(thrust::upper_bound(thrust::seq,d_keys,d_keys+NParts,currentCell->keyMax));
		currentCell->number=currentCell->cellEnd-currentCell->cellStart;
		currentCell->cellStartId=currentCell->cellStart-keys;
		currentCell->cellEndId=currentCell->cellEnd-keys;
	}
}

void __global__ mergeGrid(unsigned int* keys, unsigned int NParts,gridCell** leafs,unsigned int NLeafs,unsigned int maxParticles)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<NLeafs)
	{
		gridCell* currentCell=leafs[id];	
		gridCell* parent=currentCell->parent;
		if(parent!=NULL && currentCell->id==0)
		{
			int numberParent=0;
			bool childrenTerminal=true;
			for(int i=0;i<8;i++)
			{
				if(parent->children[i]->children[0]!=NULL)childrenTerminal=false;
				numberParent+=parent->children[i]->number;	
			}
			if(childrenTerminal && numberParent<=maxParticles)
			{
				for(int i=0;i<8;i++)
				{
					parent->children[i]->id=-1;
					parent->children[i]=NULL;
				}
				parent->number=numberParent;
				typedef unsigned long long int uint64;	
				uint64 adr;
				gridCell** pos;
				do{
					pos=leafs+(currentCell->rng.int64()%(NParts-1));
					adr=atomicCAS((uint64*)pos,(uint64)NULL,(uint64)parent);
				}while(adr!=NULL);
			}
		}
	}
}

void Grid::compactLeafs()
{
	thrust::device_ptr<gridCell*> d_leafs(leafs);
	thrust::sort(thrust::cuda::par(alloc),d_leafs,d_leafs+m_N);
	numberLeafs=thrust::raw_pointer_cast(thrust::unique(thrust::cuda::par(alloc),d_leafs,d_leafs+m_N))-leafs-1;
	gpuErrchk(cudaMemset(leafs+1+numberLeafs,0,(m_N-1-numberLeafs)*sizeof(gridCell*)));
}

void Grid::updateGrid()
{
	sortParticles();

	splitGrid<<<numberLeafs,1>>>(m_zoId,m_N,leafs+1,numberLeafs,100);
	compactLeafs();

	calculateNumber<<<numberLeafs,1>>>(m_zoId,m_N,leafs+1,numberLeafs,100);
	mergeGrid<<<numberLeafs,1>>>(m_zoId,m_N,leafs+1,numberLeafs,100);
	compactLeafs();
	removeCells<<<numberLeafs,1>>>(leafs+1,numberLeafs);
	compactLeafs();
	calculateNumber<<<numberLeafs,1>>>(m_zoId,m_N,leafs+1,numberLeafs,100);
}

void Grid::sortParticles()
{
	setZoId<<<m_N,1>>>(m_r,m_N,m_zoId,rmin,rmax);
	thrust::device_ptr<unsigned int> d_zoId(m_zoId);
	thrust::device_ptr<Segment> d_r((Segment*)m_r);
	thrust::device_ptr<Segment> d_v((Segment*)m_v);
	thrust::sort_by_key(thrust::cuda::par(alloc),d_zoId,d_zoId+m_N,make_zip_iterator(make_tuple(d_r,d_v)));	
}

gridCell* Grid::getCells()
{
	calculateNumber<<<numberLeafs,1>>>(m_zoId,m_N,leafs+1,numberLeafs,100);
	gridCell* cells=NULL;
	gpuErrchk(cudaMallocManaged(&cells,sizeof(gridCell)*numberLeafs));
	gatherCells<<<numberLeafs,1>>>(cells,numberLeafs,leafs+1);
	cudaDeviceSynchronize();
	return cells;
}
