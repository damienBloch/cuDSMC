import warnings,os
#this_dir = os.path.abspath(os.path.dirname(__file__))
#sys.path.append(os.path.join(this_dir, ""))

import numpy as np
import sympy as sym
from jinja2 import Template
from .codePrinter import CUDACodePrinter

#from .cuDSMC import cudaInit,cudaDeviceGetCount
#from .cuDSMC import DSMC as cuDSMC
#cudaInit() 

class DSMC(object):
    """Handler for Direct Simulation Monte-Carlo solver."""
    def __init__(self,r0,v0,parameters,t0:float=0,device:int=0):
        """
        Parameters
        ----------
        r0 : (3, N_T) array_like
            Array of initial positions 
        v0 : (3, N_T) array_like
            Array of initial velocities
        parameters : dictionnary
            Must contain keys "mass","number" and "cross-section" assigned to the relevant values
        t0 : float,optional
            Initial time
        device : int,optional
            Select the GPU on which to run the simulation. Make it possible to launch several simulations on the same device (assuming enough memory) or to launch simulations on several devices. The latter case should be used with multiple cpu processes, because the update on one gpu is blocking the cpu flow until the simulation has been evolved. 
        """
        self._t=t0
        #self.cuDSMC=cuDSMC(np.reshape(r0,3*len(r0[0]),order="F"),np.reshape(v0,3*len(v0[0]),order="F"),t0,device)
        self._parameters=parameters
        #self.cuDSMC.setParameters(self.parameters["mass"],self.parameters["number"],self.parameters["cross-section"])
        self._advectionTpl=""
        def setPotential(self,potential):
        """Define the external potential experienced by the particles.        
        If this method is never called, an error will be raised when running the simulation.        
        Parameters
        ----------
        potential : function
            Input potential. This should be a sympy function that takes x,y,z,t as parameters.
        """
        x,y,z,t=sym.symbols("x y z t")
        numericalEnergy=sym.N(potential(x,y,z,t))#this replaces every constant (e.g. sympy.pi) by their numerical value
        acceleration=1/self._parameters["mass"]*sym.Matrix([
                -numericalEnergy.diff(x),
                -numericalEnergy.diff(y),
                -numericalEnergy.diff(z)])
        accelSubexpr,accelFromSubexpr=sym.cse(acceleration)                    

        advectionTpl=DSMC.getAdvectionTemplate()
        codePrinter=CUDACodePrinter()
        renderedTpl = advectionTpl.render(
                numSubexpr=len(accelSubexpr),
                subexpr=[codePrinter.doprint(sub) for _,sub in accelSubexpr],
                ax=codePrinter.doprint(accelFromSubexpr[0][0]),
                ay=codePrinter.doprint(accelFromSubexpr[0][1]),
                az=codePrinter.doprint(accelFromSubexpr[0][2]))
        self._advectionTpl=renderedTpl
#        self.cuDSMC.loadPotential(rendered_tpl)
        
    def histogram(self,min_x=-np.inf,max_x=+np.inf,NX=1,min_y=-np.inf,max_y=+np.inf,NY=1,min_z=-np.inf,max_z=+np.inf,NZ=1):
        shape=()
        if(NX>1):
            shape+=(NX,)
        if(NY>1):
            shape+=(NY,)
        if(NZ>1):
            shape+=(NZ,)
        return np.reshape(self.cuDSMC.makeHistogram(min_x,max_x,NX,min_y,max_y,NY,min_z,max_z,NZ),shape,order="F")
        
    @property
    def positions(self):
        r=self.cuDSMC.getPositions()
        return np.reshape(r,(3,len(r)//3),order="F")
    @property
    def velocities(self):
        v=self.cuDSMC.getSpeeds()
        return np.reshape(v,(3,len(v)//3),order="F")
    def getPotential(self):
        return self._advectionTpl
    def advection(self,t):
        self.t=self.cuDSMC.advection(t)
        return self.t

    @classmethod
    def cudaDeviceNumber(cls):
        return cudaDeviceGetCount()
    @classmethod
    def getAdvectionTemplate(cls):
        path = os.path.join(os.path.dirname(__file__), 'advection.cu')
        return Template(open(path,"r").read(), trim_blocks=True, lstrip_blocks=True)


