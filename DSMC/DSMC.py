import numpy as np
import sympy as sym
from jinja2 import Template
from .code_printer import codePrinter
from .cuDSMC import DSMC as cuDSMC
import os

class DSMC:
    def __init__(self,r,v,t0=0):
        self.t=t0
        self.cuDSMC=cuDSMC(np.reshape(r,3*len(r[0]),order="F"),np.reshape(v,3*len(v[0]),order="F"),t0)

    def getPositions(self):
        r=self.cuDSMC.getPositions()
        return np.reshape(r,(3,len(r)//3),order="F")
    
    def getSpeeds(self):
        v=self.cuDSMC.getSpeeds()
        return np.reshape(v,(3,len(v)//3),order="F")
    def advection(self,t):
        self.t=self.cuDSMC.advection(t)
        return self.t
    
    def setPotential(self,U,parameters,a=1/12,NewtonIter=4):
        self.parameters=parameters
        x,y,z,t=sym.symbols("x y z t")
        ax,ay,az=sym.symbols("ax_tmp ay_tmp az_tmp")
        alpha,h=sym.symbols(r"alpha h")
        V=sym.N(U(x,y,z,t))
        F=sym.Matrix([-V.diff(x),-V.diff(y),-V.diff(z)])/parameters["mass"]
        f=sym.Matrix([ax,ay,az])-F.subs({x:x+alpha*h**2*ax,y:y+alpha*h**2*ay,z:z+alpha*h**2*az})
        Jf=sym.zeros(3,3)
        Jf[0]=f.diff(ax)
        Jf[1]=f.diff(ay)
        Jf[2]=f.diff(az)
        A = sym.Matrix(3, 3, sym.symbols('A:3:3'))
        Jf_inv=(A.adjugate()/A.det()).subs(dict(zip(list(A), list(Jf))))
        a_new=sym.Matrix([ax,ay,az])-Jf_inv*f
        s=sym.cse(a_new) 
        s_energy=sym.cse(V)
        path = os.path.join(os.path.dirname(__file__), 'advection.cu')
        tpl = Template(open(path,"r").read())
        cp=codePrinter()
        rendered_tpl = tpl.render(
            subexpression_number=len(s[0]),
            subexpressions=[cp.doprint(s[0][i][1]) for i in range(len(s[0]))],
            ax=cp.doprint(s[1][0][0]),ay=cp.doprint(s[1][0][1]),az=cp.doprint(s[1][0][2]),
            NewtonIter=NewtonIter,alpha=a,
            subexpression_number_energy=len(s_energy[0]),
            subexpressions_energy=[cp.doprint(s_energy[0][i][1]) for i in range(len(s_energy[0]))],
            energy=cp.doprint(s_energy[1][0]))
        self.advection_tpl=rendered_tpl
        self.subexpressions=s
        self.cuDSMC.setParameters(parameters["mass"],parameters["number"],parameters["cross-section"])
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
        
