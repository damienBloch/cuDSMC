# cuDSMC
Simple CUDA implementation of the DSMC method

## Status

- [x] Monoatomic single specie 3D DSMC simulation
- [x] Python wrapper to initialise simulations and request results from CUDA core
- [x] Arbitrary external potential. Defined in Python, compiled and loaded on the GPU at runtime using the CUDA driver API
- [x] Basic AMR on GPU based on particles density
- [ ] Adaptative timestep
- [ ] Multiple species
- [ ] Internal degree freedoms
