# cuDSMC
Simple CUDA implementation of the DSMC method.
This code was written to compare with some cold atoms experiments (thermalisation, evaporation, ...). In these situations, collisions are easier to handle (simple interactions, no complex chemistry) but the external potential depends on the experimental implementation. 
For ease of manipulation, the simulation can be fully set up from a Python interface, while still providing the higher performances of CUDA enabled GPUs. 

## Status

- [x] Monoatomic single specie 3D DSMC simulation
- [x] Python wrapper to initialise simulations and request results from CUDA core
- [x] Arbitrary external potential. Defined in Python, transcripted in CUDA code with sympy, compiled and loaded on the GPU at runtime using the CUDA driver API
- [x] Basic AMR on GPU based on particles density using an octree structure
- [ ] Lower dimensionality
- [ ] Particle inflows/outflows
- [ ] Adaptative timestep
- [ ] Multiple species

## Prerequisites

- [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (version 3.0 or higher)
- [NumPy](https://numpy.org/)
- [Sympy](https://www.sympy.org/en/index.html)

## How to use

Download from github 

```
git clone https://github.com/damienBloch/cuDSMC
```

Create build folder

```
cd cuDSMC
mkdir obj
```

Compile C++ and CUDA sources (requires nvcc and CUDA toolkit)

```
cd src
make -j4
```

This will create a python package that can used be like any other package, either by placing it in the same directory as the python script or by installing it with pip

```
cd ..
pip install DSMC
```


