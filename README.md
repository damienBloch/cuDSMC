<meta name="google-site-verification" content="huQ-v4YHMkRx75PQ8RDl41gF7mfMoqRamn8ih8EuRXk" />
# cuDSMC
Simple CUDA implementation of the DSMC method

## Status

- [x] Monoatomic single specie 3D DSMC simulation
- [x] Python wrapper to initialise simulations and request results from CUDA core
- [x] Arbitrary external potential. Defined in Python, compiled and loaded on the GPU at runtime using the CUDA driver API
- [x] Basic AMR on GPU based on particles density
- [ ] Adaptative timestep
- [ ] Multiple species

## Prerequisites

- [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (version 3.0 or higher)
- [NumPy](https://numpy.org/)
- [Sympy](https://www.sympy.org/en/index.html)
- [Jinja2](https://jinja.palletsprojects.com/en/2.11.x/)
- [pybind11](https://github.com/pybind/pybind11)

