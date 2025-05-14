import torch
import numpy as np

class M_torch:
    def __init__(self, Nc, eps, phi, wl, I, E0, res, device='cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        self.T = Nc * 2.0 * torch.pi / wl
        self.A0 = torch.sqrt(torch.tensor(I, device=device)) / wl
        self.eps = eps
        self.wl = wl
        self.phi = phi
        self.E0 = E0
        self.ts = torch.linspace(0.0, self.T, steps=res, device=device)
        vector = torch.stack([
            torch.zeros_like(self.ts),
            self.eps * torch.sin(self.wl * self.ts + self.phi),
            torch.cos(self.wl * self.ts + self.phi)
        ])
        self.As = (torch.sin(torch.pi * self.ts / self.T) ** 2) * self.A0 / torch.sqrt(torch.tensor(1 + self.eps ** 2, device=device, dtype=torch.float32)) * vector

    def f_phi_0(self, k):
        # k: (N, 3)
        k_mag = torch.norm(k, dim=-1)
        return 2 * torch.sqrt(torch.tensor(2.0, device=self.device)) / torch.pi * (1 / (k_mag ** 2 + 1) ** 2)

    def exp_integrand(self, k):
        # k: (N, 3)
        # As: (3, T)
        # We want to broadcast k to (N, T, 3)
        N = k.shape[0]
        T = self.ts.shape[0]
        k_expanded = k[:, None, :]  # (N, 1, 3)
        AsT = self.As.T[None, :, :]  # (1, T, 3)
        kA = k_expanded + AsT  # (N, T, 3)
        return 1j * 0.5 * torch.sum(kA * kA, dim=-1)  # (N, T)

    def exp_integral(self, k):
        # k: (N, 3)
        E0_term = self.E0 * self.ts  # (T,)

        exp_ys = self.exp_integrand(k) # (N, T)
        dt = self.ts[1] - self.ts[0]
        # Cumulative sum for integration

        integral = torch.cumulative_trapezoid(exp_ys, self.ts, dim=-1)
        # Prepend a zero along the integration axis to match SciPy's 'initial=0'
        pad_shape = list(integral.shape)
        pad_shape[-1] = 1
        integral = torch.cat([torch.zeros(pad_shape, dtype=integral.dtype, device=integral.device), integral], dim=-1)
        # Subtract 1j * E0 * t for each t
        result = torch.exp(1j * (integral - E0_term))  # (N, T)
        return result  # (N, T)

    def integrands(self, k):
        # k: (N, 3)
        # A: (3, T) vector potential
        N = k.shape[0]
        T = self.ts.shape[0]
        
        # Get exponential phase factor
        exp_int = self.exp_integral(k)  # (N, T)
        
        # Properly broadcast for efficient multiplication:
        # 1. Transpose A to (T, 3) and expand to (1, T, 3)
        As_T = self.As.permute(1, 0)[None, :, :]  # (1, T, 3)
        
        # 2. Expand exp_int to (N, T, 1)
        exp_int_expanded = exp_int[:, :, None]  # (N, T, 1)
        
        # 3. Multiply with broadcasting
        # This gives (N, T, 3)
        result = exp_int_expanded * As_T  # (N, T, 3)
        
        # 4. Permute back to (N, 3, T) for consistency with the rest of the code
        return result.permute(0, 2, 1)  # (N, 3, T)

    def integral(self, k):
        # k: (N, 3)
        ys = self.integrands(k)  # (N, 3, T)
        integral_vec = torch.trapz(ys, self.ts, dim=-1)  # (N, 3), likely complex
        # Ensure k is the same dtype as integral_vec
        # result = torch.einsum('ij,ij->i', k.to(integral_vec.dtype), integral_vec)  # (N,)
        # result = torch.sum(k * integral_vec, dim=-1)
        result = torch.einsum('ij,ij->i', k.to(integral_vec.dtype), integral_vec)
        return result

    def Mk0(self, k_values):
        # k_values: (N, 3)
        f_phi_0s = self.f_phi_0(k_values)  # (N,)
        integrals = self.integral(k_values)  # (N,)
        Mk0s = -1j * f_phi_0s * integrals  # (N,)
        return Mk0s

    def Mk0_squared(self, k_values):
        Mk0s = self.Mk0(k_values)
        #normalize Mk0s
        # Mk0s = Mk0s / torch.norm(Mk0s, dim=0)
        return Mk0s.abs() ** 2