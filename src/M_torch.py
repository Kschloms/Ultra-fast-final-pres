import torch
import numpy as np

class M_torch:
    def __init__(self, Nc, eps, phi, wl, I, E0, res, device='cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        self.T = Nc * 2.0 * torch.pi / wl
        self.A0 = torch.sqrt(torch.tensor(I, device=device, dtype=torch.float64)) / torch.tensor(wl, dtype=torch.float64, device=device)
        self.eps = eps
        self.wl = wl
        self.phi = phi
        self.E0 = E0
        self.ts = torch.linspace(0.0, self.T, steps=res, device=device, dtype=torch.float64)
        # Vector potential (3, T)
        vector = torch.stack([
            torch.zeros_like(self.ts),
            self.eps * torch.sin(self.wl * self.ts + self.phi),
            torch.cos(self.wl * self.ts + self.phi)
        ])
        self.As = (torch.sin(torch.pi * self.ts / self.T) ** 2) * self.A0 / torch.sqrt(torch.tensor(1 + self.eps ** 2, device=device)) * vector

    def f_phi_0(self, k):
        k = k.to(dtype=torch.float64)
        k_mag = torch.norm(k, dim=-1)
        return 2 * torch.sqrt(torch.tensor(2.0, device=self.device)) / torch.pi * (1 / (k_mag ** 2 + 1) ** 2)

    def exp_integrand(self, k):
        # k: (N, 3), As: (3, T)
        k = k.to(dtype=torch.complex128)
        AsT = self.As.T.to(dtype=torch.complex128)  # (T, 3)
        kA = k[:, None, :] + AsT[None, :, :]        # (N, T, 3)
        return 1j * 0.5 * torch.sum(kA * kA, dim=-1)  # (N, T)

    def exp_integral(self, k):
        # Exponential phase factor for each k and t
        exp_ys = self.exp_integrand(k)  # (N, T)
        # Integrate over time axis
        integral = torch.cumulative_trapezoid(exp_ys, self.ts, dim=-1)
        # Prepend zero to match scipy's initial=0
        integral = torch.cat([torch.zeros((*integral.shape[:-1], 1), dtype=integral.dtype, device=integral.device), integral], dim=-1)
        phase = integral - 1j * self.E0 * self.ts  # (N, T)
        return torch.exp(phase)  # (N, T)

    def integrands(self, k):
        # Returns (N, 3, T)
        exp_int = self.exp_integral(k)  # (N, T)
        As = self.As.to(dtype=exp_int.dtype)  # (3, T)
        # Broadcasting: (N, 1, T) * (1, 3, T) -> (N, 3, T)
        return  As[None, :, :] *  exp_int[:, None, :]

    def integral(self, k):
        # Integrate over time axis using trapz for stability
        ys = self.integrands(k)  # (N, 3, T)
        integral_vec = torch.trapz(ys, self.ts, dim=-1)  # (N, 3)
        # Dot product with k
        return torch.einsum('ij,ij->i', k.to(integral_vec.dtype), integral_vec)

    def Mk0(self, k):
        f_phi_0s = self.f_phi_0(k)  # (N,)
        integrals = self.integral(k)  # (N,)
        return -1j * f_phi_0s * integrals  # (N,)

    def Mk0_squared(self, k):
        Mk0s = self.Mk0(k)
        return Mk0s.abs() ** 2