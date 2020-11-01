extern "C" __device__ void computeAcceleration(double x, double y, double z, double t, 
double* ax, double *ay, double* az)
{
	{% for i in range(numSubexpr) %}
	double x{{i}} = {{subexpr[i]}};
	{% endfor %}

	*ax = {{ax}};
	*ay = {{ay}};
	*az = {{az}};
}

extern "C" __global__ void advectionKernel(double* r, double* v, unsigned int N,double dt,double t)
{
	//This kernel implements Verlet integration, which is one of the simplest symplectic explicit method
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<N)
	{
		double rx=r[3*id],ry=r[3*id+1],rz=r[3*id+2];
		double vx=v[3*id],vy=v[3*id+1],vz=v[3*id+2];
		double ax=0,ay=0,az=0;
	
		rx+=dt/2*vx;
		ry+=dt/2*vy;
		rz+=dt/2*vz;
		
		computeAcceleration(rx,ry,rz,t,&ax,&ay,&az);

		vx+=dt*ax;
		vy+=dt*ay;
		vz+=dt*az;

		rx+=dt/2*vx;
		ry+=dt/2*vy;
		rz+=dt/2*vz;

		r[3*id+0]=rx;
		r[3*id+1]=ry;
		r[3*id+2]=rz;

		v[3*id+0]=vx;
		v[3*id+1]=vy;
		v[3*id+2]=vz;
	}
}
