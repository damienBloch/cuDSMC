#ifndef HISTOGRAM_H
#define HISTOGRAM_H
#include"utils.h"
#include<thrust/sort.h>
#include<thrust/binary_search.h>
#include<thrust/adjacent_difference.h>
#include<thrust/execution_policy.h>
class Histogram
{
	public:
		Histogram(double min_x,double max_x, int NX,double min_y,double max_y, int NY,double min_z,double max_z, int NZ);
		thrust::device_vector<unsigned int>& bin(thrust::device_vector<double> r);
	private:
		double min_x,min_y,min_z,max_x,max_y,max_z;
		int NX,NY,NZ;
		thrust::device_vector<unsigned int> mHist;
};
#endif
