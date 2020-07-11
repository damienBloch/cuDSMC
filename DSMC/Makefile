objects = DSMC.o nvrtc.o histogram.o grid.o collision.o

all: $(objects)
	nvcc -arch=sm_50 -shared $(objects) -o cuDSMC`python3-config --extension-suffix` -lcudart -lcuda -lnvrtc -lcudadevrt

%.o: %.cpp
	g++ -c -fPIC `python3 -m pybind11 --includes` -std=c++11 $< -o $@

%.o: %.cu
	nvcc -arch=sm_50 -Xcompiler '-fPIC' -dc -c $< -o $@

clean:
	rm -f *.o 
