import os,sys,warnings
this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_dir, ""))

import numpy as np
import sympy as sym
from jinja2 import Template
from .code_printer import codePrinter

from .cuDSMC import cudaInit,cudaDeviceGetCount
from .cuDSMC import DSMC as cuDSMC
cudaInit() 
       

class DSMC(object):
    """Handler for Direct Simulation Monte-Carlo solver."""
    @classmethod
    def cudaDeviceNumber(cls):
        return cudaDeviceGetCount()
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
        """
        self.t=t0
        self.cuDSMC=cuDSMC(np.reshape(r0,3*len(r0[0]),order="F"),np.reshape(v0,3*len(v0[0]),order="F"),t0,device)
        self.parameters=parameters
        self.cuDSMC.setParameters(self.parameters["mass"],self.parameters["number"],self.parameters["cross-section"])
        self.advection_tpl=""
    @property
    def positions(self):
        r=self.cuDSMC.getPositions()
        return np.reshape(r,(3,len(r)//3),order="F")

    def getPositions(self):
        r=self.cuDSMC.getPositions()
        return np.reshape(r,(3,len(r)//3),order="F")
    def getPotential(self):
        return self.advection_tpl
    
    def getSpeeds(self):
        v=self.cuDSMC.getSpeeds()
        return np.reshape(v,(3,len(v)//3),order="F")
    def advection(self,t):
        self.t=self.cuDSMC.advection(t)
        return self.t
    
    def setPotential(self,potential,method="Verlet"):
        """Define the external potential experienced by the particles.
        
        If this method is never called, an error will be raised when runing the simulation.
        
        Parameters
        ----------
        potential : function
            Input potential
        method : str, optional
            Integration method. Methods other than Verlet are implicit and have a longer overhead time.
            - "Verlet",default. Cheap explicit integrator.
            - "Implicit midpoint". Euler midpoint method.
            - "Störmer-Cowel". Fourth order implicit solver.
            - "Stable implicit".
        """
        try:
            a={"Verlet":0,"Implicit midpoint":1/4,"Störmer-Cowel":1/12,"Stable implicit":1/3}[method]
        except:
            warnings.warn("No method "+method+" available. Falling back to Verlet method.")
            method="Verlet"
            a=0     
        if(method=="Verlet"):
            NewtonIter=0
        else:
            NewtonIter=3        
        x,y,z,t=sym.symbols("x y z t")
        V=sym.N(potential(x,y,z,t))
        F=sym.Matrix([-V.diff(x),-V.diff(y),-V.diff(z)])/self.parameters["mass"]
        cp=codePrinter()
        path = os.path.join(os.path.dirname(__file__), 'advection.cu')
        tpl = Template(open(path,"r").read())
        s_energy=sym.cse(V) 
        if(a>0):        
            ax,ay,az=sym.symbols("ax_tmp ay_tmp az_tmp")
            alpha,h=sym.symbols(r"alpha h")
            f=sym.Matrix([ax,ay,az])-F.subs({x:x+alpha*h**2*ax,y:y+alpha*h**2*ay,z:z+alpha*h**2*az})
            Jf=sym.zeros(3,3)
            Jf[0]=f.diff(ax)
            Jf[1]=f.diff(ay)
            Jf[2]=f.diff(az)
            A = sym.Matrix(3, 3, sym.symbols('A:3:3'))
            Jf_inv=(A.adjugate()/A.det()).subs(dict(zip(list(A), list(Jf))))
            a_new=sym.Matrix([ax,ay,az])-Jf_inv*f
            s=sym.cse(a_new)       
        elif(a==0):
            s=sym.cse(F)                    
        rendered_tpl = tpl.render(
                subexpression_number=len(s[0]),
                subexpressions=[cp.doprint(s[0][i][1]) for i in range(len(s[0]))],
                ax=cp.doprint(s[1][0][0]),ay=cp.doprint(s[1][0][1]),az=cp.doprint(s[1][0][2]),
                NewtonIter=NewtonIter,alpha=a,
                subexpression_number_energy=len(s_energy[0]),
                subexpressions_energy=[cp.doprint(s_energy[0][i][1]) for i in range(len(s_energy[0]))],
                energy=cp.doprint(s_energy[1][0]))   
        self.advection_tpl=rendered_tpl
        self.cuDSMC.loadPotential(rendered_tpl)
        
    def histogram(self,min_x=-np.inf,max_x=+np.inf,NX=1,min_y=-np.inf,max_y=+np.inf,NY=1,min_z=-np.inf,max_z=+np.inf,NZ=1):
        shape=()
        if(NX>1):
            shape+=(NX,)
        if(NY>1):
            shape+=(NY,)
        if(NZ>1):
            shape+=(NZ,)
        return np.reshape(self.cuDSMC.makeHistogram(min_x,max_x,NX,min_y,max_y,NY,min_z,max_z,NZ),shape,order="F")
        
