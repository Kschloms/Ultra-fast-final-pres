import torch


class M_torch:
    def __init__(self, Nc, eps, phi, wl, I, E0, res, device='cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        self.T = Nc * 2 * torch.pi / wl
        self.A0 = torch.sqrt(torch.tensor(I, device=device)) / wl
        self.eps = eps
        self.wl = wl
        self.phi = phi
        self.E0 = E0
        self.ts = torch.linspace(0, self.T, steps=res, device=device)
        vector = torch.stack([
            torch.zeros_like(self.ts),
            self.eps * torch.sin(self.wl * self.ts + self.phi),
            torch.cos(self.wl * self.ts + self.phi)
        ])
        self.As = (torch.sin(torch.pi * self.ts / self.T) ** 2) * self.A0 / torch.sqrt(torch.tensor(1 + self.eps ** 2, device=device, dtype=torch.float32)) * vector

    def f_phi_0(self, k):
        k_mag = torch.norm(k, dim=-1)
        return 2 * torch.sqrt(torch.tensor(2.0, device=self.device)) / torch.pi * (1 / (k_mag ** 2 + 1) ** 2)

    def exp_integrand(self, k):
        # k: (..., 3)
        # As: (3, N)
        # Need to broadcast k to (N, 3) for each time step
        k = k.unsqueeze(-2)  # (..., 1, 3)
        AsT = self.As.T  # (N, 3)
        kA = k + AsT  # (..., N, 3)
        return 1j * 0.5 * torch.sum(kA * kA, dim=-1)  # (..., N)

    def exp_integral(self, k):
        exp_ys = self.exp_integrand(k)  # (..., N)
        # Implement cumulative trapezoid or just definite integral
        dt = self.ts[1] - self.ts[0]
        integral = torch.cumsum(exp_ys, dim=-1) * dt
        return torch.exp(integral - 1j * self.E0 * self.ts)

    def integrands(self, k):
        return self.As * self.exp_integral(k)  # (3, N)

    def integral(self, k):
        ys = self.integrands(k)  # (3, N)
        dt = self.ts[1] - self.ts[0]
        integral_vec = torch.trapz(ys, self.ts, dim=-1)  # (3,), likely complex
        return torch.dot(k.to(integral_vec.dtype), integral_vec)

    def Mk0(self, k_values):
        # k_values: (N, 3)
        f_phi_0s = self.f_phi_0(k_values)
        integrals = torch.stack([self.integral(k) for k in k_values])
        Mk0s = 1j * f_phi_0s * integrals
        return Mk0s

    def Mk0_squared(self, k_values):
        Mk0s = self.Mk0(k_values)
        return torch.abs(Mk0s) ** 2