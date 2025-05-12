# -*- coding: utf-8 -*-
"""
Created on Wed May  7 11:38:36 2025

@author: Frej
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.integrate as integrate
from concurrent.futures import ProcessPoolExecutor

    

class M:
    def __init__(self,Nc,eps,phi,wl,I,E0,res):
        self.T = Nc*2*np.pi/wl
        self.A0 = np.sqrt(I)/wl
        
        self.eps = eps
        self.wl = wl
        self.phi = phi
        self.E0 = E0
        
        #Fields and times
        self.ts = np.linspace(0,self.T,num=res)
        vector = np.asarray([self.ts*0,self.eps*np.sin(self.wl*self.ts+self.phi),np.cos(self.wl*self.ts+self.phi)])
        self.As = np.sin(np.pi*self.ts/self.T)**2*self.A0/((1+self.eps**2)**(1/2))*vector

    
    #Defining Exponential
    def exp_integrand(self,k):
        return [1j * (1/2) * ((k+A) @ (k+A)) for A in self.As.T]   
    def exp_integral(self,k):
        exp_ys = self.exp_integrand(k)
        return np.e**(integrate.cumulative_trapezoid(exp_ys,self.ts,initial=0)-1j * self.E0*self.ts)
    #Defining Integral
    def integrands(self,k):
        return self.As * self.exp_integral(k)
    def integral(self,k):
        ys = self.integrands(k)  # shape (3, N)
        # Integrate each component over time, get shape (3,)
        integral_vec = integrate.cumulative_trapezoid(ys, self.ts, axis=1, initial=0)[:, -1]  # shape (3,)
        return np.dot(k, integral_vec)
    
    #Defining multiplication with fourier transform
    def f_phi_0(self,k):
        k_mag = np.linalg.norm(k)
        return 2*2**(1/2)/np.pi*(1/(k_mag**2+1)**2)
    def Mk0(self, k_values):
        """
        Calculate Mk0 for an array of k values
        k_values: numpy array of shape (n, 3) where each row is a k-vector
        """
        k_values = np.atleast_2d(k_values)
        f_phi_0s = np.apply_along_axis(self.f_phi_0, 1, k_values)
        integrals = np.apply_along_axis(self.integral, 1, k_values)
        Mk0s = 1j * f_phi_0s * integrals
        print(Mk0s.shape)
        return Mk0s

    # Test with parallelization
    # def Mk0(self, k_values):
    #     k_values = np.atleast_2d(k_values)
    #     with ProcessPoolExecutor() as executor:
    #         f_phi_0s = list(executor.map(self.f_phi_0, k_values))
    #         integrals = list(executor.map(self.integral, k_values))
    #     Mk0s = 1j * np.array(f_phi_0s) * np.array(integrals)
    #     return Mk0s

    def Mk0_squared(self, k_values):
        """
        Calculate |Mk0|^2 for an array of k values
        k_values: numpy array of shape (n, 3) where each row is a k-vector
        """
        results = self.Mk0(k_values)
        return np.array([np.real(np.vdot(mk, mk)) for mk in results])


# %%

# %%

#%%
if __name__ == "__main__":
    # Field parameters
    Nc = 5
    eps = 0
    phi = 0
    wl = 0.057
    I = 0.003
    E0 = -0.500
    res = 1000

    M1 = M(Nc, eps, phi, wl, I, E0, res)

    kys = np.linspace(-2, 2, num=50)
    kzs = np.linspace(-2, 2, num=50)
    kys_grid, kzs_grid = np.meshgrid(kys, kzs, indexing='ij')
    ks = np.stack([np.zeros_like(kys_grid), kys_grid, kzs_grid], axis=-1).reshape(-1, 3)

    Mk_squared_vals = M1.Mk0_squared(ks)
    Mk_squared_vals = Mk_squared_vals.reshape(len(kys), len(kzs))

    plt.contourf(kys, kzs, Mk_squared_vals, 50)
    plt.xlabel(r'$K_y$')
    plt.ylabel(r'$K_z$')
    plt.colorbar()
    plt.show()



