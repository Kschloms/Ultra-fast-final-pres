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

    def get_A(self):
        return self.As
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
        exp_ys = self.exp_integrand(k)  # (N, T)
        dt = self.ts[1] - self.ts[0]
        # Cumulative sum for integration
        integral = torch.cumsum(exp_ys, dim=-1) * dt  # (N, T)
        # Subtract 1j * E0 * t for each t
        E0_term = 1j * self.E0 * self.ts  # (T,)
        result = torch.exp(integral - E0_term)
        return result  # (N, T)

    def integrands(self, k):
        # k: (N, 3)
        # As: (3, T)
        # exp_integral: (N, T)
        exp_int = self.exp_integral(k)  # (N, T)
        # As: (3, T) -> (1, 3, T)
        As_exp = self.As[None, :, :]  # (1, 3, T)
        # exp_int: (N, T) -> (N, 1, T)
        exp_int_exp = exp_int[:, None, :]  # (N, 1, T)
        # Multiply and get (N, 3, T)
        return As_exp * exp_int_exp  # (N, 3, T)

    def integral(self, k):
        # k: (N, 3)
        ys = self.integrands(k)  # (N, 3, T)
        dt = self.ts[1] - self.ts[0]
        # Integrate over T (time axis)
        integral_vec = torch.trapz(ys, self.ts, dim=-1)  # (N, 3)
        # Dot product for each k: sum over last dim
        result = torch.sum(k * integral_vec, dim=-1)  # (N,)
        return result

    def Mk0(self, k_values):
        # k_values: (N, 3)
        f_phi_0s = self.f_phi_0(k_values)  # (N,)
        integrals = self.integral(k_values)  # (N,)
        Mk0s = 1j * f_phi_0s * integrals  # (N,)
        return Mk0s

    def Mk0_squared(self, k_values):
        Mk0s = self.Mk0(k_values)
        return torch.abs(Mk0s) ** 2