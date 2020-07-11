extern "C" __device__ void NewtonStep(double x, double y, double z, double *ax, double *ay, double *az,double alpha, double h, double t)
{
	double ax_tmp=*ax;
	double ay_tmp=*ay;
	double az_tmp=*az;
	{% for i in range(subexpression_number) %}{% set subexpression = subexpressions[i]%}
	double x{{i}} = {{subexpression}};{% endfor %}
	*ax = {{ax}};
	*ay = {{ay}};
	*az = {{az}};
}

extern "C" __global__ void advectionKernel(double* r, double* v, unsigned int N,double dt,double t)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<N)
	{
		double rx=r[3*id],ry=r[3*id+1],rz=r[3*id+2];
		double vx=v[3*id],vy=v[3*id+1],vz=v[3*id+2];
		double ax=0,ay=0,az=0;
	
		rx+=dt/2*vx;
		ry+=dt/2*vy;
		rz+=dt/2*vz;

		NewtonStep(rx,ry,rz,&ax,&ay,&az,0,dt,t+dt/2);
		for(int i=0;i<{{NewtonIter}};i++)
			NewtonStep(rx,ry,rz,&ax,&ay,&az,{{alpha}},dt,t+dt/2);

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

extern "C" __device__ double energyCalculation(double x, double y, double z, double t)
{
	{% for i in range(subexpression_number_energy) %}{% set subexpression = subexpressions_energy[i]%}
	double x{{i}} = {{subexpression}};{% endfor %}
	double e={{energy}};
	return e;
}

extern "C" __global__ void computeEnergy(double *r, double *v,double *energy,unsigned int N, double t, double mass)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;	
	if(id<N)
	{
		double e=0;
		e+=energyCalculation(r[3*id+0],r[3*id+1],r[3*id+2],t);
		e+=mass/2*v[3*id+0]*v[3*id+0];
		e+=mass/2*v[3*id+1]*v[3*id+1];
		e+=mass/2*v[3*id+2]*v[3*id+2];
		energy[id]=e;
	}
}
