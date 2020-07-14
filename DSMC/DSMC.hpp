#ifndef DSMC_H
#define DSMC_H
#include"utils.h"
#include"nvrtc.hpp"
#include"histogram.h"
#include"grid.hpp"
#include"pybind11/pybind11.h"
#include"pybind11/numpy.h"
namespace py = pybind11;
class DSMC {
	public:
		DSMC(py::array_t<double>& r,py::array_t<double>& v,double t0);
		double advection(double t);
		py::array_t<double> getPositions();
		py::array_t<double> getSpeeds();
		~DSMC();
		void loadPotential(char *str);
		double update(double);
		double getTime(){return m_t;};
		py::array_t<gridCell> getGrid();
		py::array_t<unsigned int> makeHistogram(double min_x,double max_x, int NX,double min_y,double max_y, int NY,double min_z,double max_z, int NZ);	
		void energy();
		void setParameters(double mass,double number, double cross_section)
		{m_m=mass;m_NP=number;m_cs=cross_section;};

	private:
		double m_t;
		unsigned int m_NT;
		double *m_r, *m_v;
		struct module{
			bool moduleLoaded=false;
			CUmodule module;
			CUfunction kernel;
			int minGridSize,blockSize;
		};
		CUfunction m_energyKernel;
		module m_advection;
		CUcontext m_cuContext;
		Grid *m_grid=NULL;
		double *m_energy=NULL;
		double m_m,m_NP,m_cs;
};


PYBIND11_MODULE(cuDSMC, m) {
	py::class_<DSMC>(m, "DSMC")
		.def(py::init<py::array_t<double>&,py::array_t<double>&,double>())
		.def("getPositions", &DSMC::getPositions,py::return_value_policy::take_ownership)
		.def("getSpeeds", &DSMC::getSpeeds,py::return_value_policy::take_ownership)
		.def("advection", &DSMC::advection)
		.def("update", &DSMC::update)
		.def("getTime", &DSMC::getTime)
		.def("setParameters", &DSMC::setParameters)
		.def("energy", &DSMC::energy)
		.def("loadPotential", &DSMC::loadPotential)
		.def("getGrid",&DSMC::getGrid,py::return_value_policy::take_ownership)
		.def("makeHistogram", &DSMC::makeHistogram,py::return_value_policy::take_ownership);
		PYBIND11_NUMPY_DTYPE(double3, x, y,z);
		PYBIND11_NUMPY_DTYPE(gridCell, number, level,rmin,rmax,cellStartId,cellEndId,id,maxVr);
}

#endif
