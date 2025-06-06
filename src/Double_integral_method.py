# -*- coding: utf-8 -*-
"""
Created on Wed May  7 11:38:36 2025

@author: Frej
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.integrate as integrate


    

class M:
    def __init__(self,Nc,eps,phi,wl,I,E0):
        self.T = Nc*2*np.pi/wl
        self.A0 = np.sqrt(I)/wl
        
        self.eps = eps
        self.wl = wl
        self.phi = phi
        self.E0 = E0
    
    #Defining Field
    def f(self,t):
        return np.sin(np.pi*t/self.T)**2
    def A(self,t):
        vector = np.asarray([0,self.eps*np.sin(self.wl*t+self.phi),np.cos(self.wl*t+self.phi)])
        return self.f(t)*self.A0/(1+self.eps**2)**(1/2)*vector
    
    #Defining Exponential
    def exp_integrand(self, t, k):
        kA = k + self.A(t)
        return 1j * 0.5 * np.dot(kA, kA)
    def exp_integral(self, t, k):
        return np.exp(integrate.quad_vec(self.exp_integrand, 0, t, args=(k,))[0] - 1j * self.E0 * t)
    #Defining Integral
    def integrands(self, t, k):
        return self.A(t) * self.exp_integral(t, k)
    def integral(self, k):
        result = integrate.quad_vec(self.integrands, 0, self.T, args=(k,))[0]
        return np.dot(k, result)
    
    #Defining multiplication with fourier transform
    def f_phi_0(self,k):
        k_mag = np.linalg.norm(k)
        return 2*2**(1/2)/np.pi*(1/(k_mag**2+1)**2)
    def Mk0(self, k_values):
        """
        Calculate Mk0 for an array of k values
        k_values: numpy array of shape (n, 3) where each row is a k-vector
        """
        Mk0s = []
        for k_vec in k_values:
            Mk = 1j * self.f_phi_0(k_vec) * self.integral(k_vec)
            Mk0s.append(Mk)
        return np.array(Mk0s)

    def Mk0_squared(self, k_values):
        results = self.Mk0(k_values)
        return np.array([np.real(np.vdot(mk, mk)) for mk in results])

# %%

# %%

#%%

if __name__ == "__main__":
    #Field parameters
    Nc = 2 #Field revolutions in envelope
    eps = 0 #(-1,0,1), Polarization of light
    phi = 0 #Phase difference between envelope and field
    #all values in au
    wl = 0.057 #800nm wavelength
    I = 0.003 #I=10^14 w/cm^2  
    E0 = 0.500 #13.6 eV
    

    M1 = M(Nc,eps,phi,wl,I,E0)

    #Trial calculation for k along the z-axis
    for k in np.linspace(0,1,num=50):
        print(M1.Mk0_squared([0,0,k]))



# %%
