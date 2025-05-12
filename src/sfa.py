import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate

class Laser:
    def __init__(self, name: str, wavelength: float, intensity: float, num_cycles_in_pulse: float, polarization : float = 0.0):
        self.name = name
        self.wavelength = wavelength  # in nm
        self.frequency = 3e8 / (self.wavelength * 1e-9)
        # Convert frequency to atomic units (1 a.u. = 2.418884e-17 s)
        self.frequency_au = self.frequency * 2.418884e-17
        self.intensity = intensity  # in W/cm^2
        # Convert intensity from W/cm^2 to atomic units (1 a.u. = 3.50944758 × 10^16 W/cm^2)
        self.intensity_au = self.intensity / 3.50944758e16
        self.pulse_duration = num_cycles_in_pulse * 2 * np.pi / self.frequency_au  # in au
        self.epsilon = polarization 

    def __str__(self):
        return f"Laser {self.name}: {self.wavelength} nm, {self.intensity} W/cm^2, {self.pulse_duration} fs"
    
    
    def _envelope(self, t: np.ndarray) -> np.ndarray:
        # Sin^2 envelope function for the laser pulse
        # Sin² envelope function for the laser pulse
        envelope = np.zeros_like(t)
        envelope[(t >= 0) & (t <= self.pulse_duration)] = (np.sin(np.pi * t[(t >= 0) & (t <= self.pulse_duration)] / self.pulse_duration)) ** 2
        return envelope
    
    def plot_envelope(self, ax: plt.axes, t: np.ndarray) -> None:
        envelope = self._envelope(t)
        ax.plot(t, envelope, label='Envelope', color='blue')
        ax.set_title(f"Envelope of {self.name} Laser Pulse")
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.legend()
        ax.grid()

    def A(self, t, phase: float = np.pi) -> np.ndarray:
        A_0 = np.sqrt(self.intensity_au) / self.frequency_au
        A_x = np.zeros_like(t)  # x-component is zero for linearly polarized light
        A_y = A_0 * self._envelope(t) * self.epsilon * np.sin(self.frequency_au * t + phase) / np.sqrt(1 + self.epsilon**2)
        A_z = A_0 * self._envelope(t) * np.cos(self.frequency_au * t + phase) / np.sqrt(1 + self.epsilon**2)
        return np.array([A_x, A_y, A_z])

    def plot_electric_field(self, ax: plt.axes, phase: float = 0.0) -> None:
        t = np.linspace(0, self.pulse_duration, 1000)
        A = self.A(t, phase)
        ax.plot(t, A[1], label='Electric Field (y-component)', color='blue')
        ax.plot(t, A[2], label='Electric Field (z-component)', color='red')
        ax.set_title(f"Electric Field of {self.name} Laser Pulse")
        ax.set_xlabel("Time (a.u.)")
        ax.set_ylabel("Electric Field (a.u.)")
        ax.legend()
        ax.grid()

    

class Photo_Electron:
    def __init__(self, energy: float, k_x, k_y, k_z):
        self.energy = energy  # in eV
        self.k = np.array([k_x, k_y, k_z])

    def __str__(self):
        return f"Photoelectron: {self.energy} eV, {self.momentum} kg*m/s"

class SFA:
    def __init__(self, laser: Laser, ks : np.array, photoelectrons: np.array = None):
        self.laser = laser
        self.photoelectrons = photoelectrons
        if photoelectrons is not None:
            self.ks = np.array([pe.k for pe in photoelectrons])
        else:
            self.ks = ks

    def __str__(self):
        return f"SFA with {self.laser} and {self.photoelectrons}"    

    def ground_state_wf(self, k) -> np.array:
        # Ground state wave function in momentum space (Gaussian)
        return np.exp(-k**2 / 2)  # Normalized Gaussian wave function

    def ground_state_energy(self, k) -> float:
        # Ground state energy in atomic units (1 a.u. = 27.2114 eV)
        return k**2 / 2
    
    @property
    def M_SFA_VG(self) -> np.array:
        # Create a time array for the integration
        t_arr = np.linspace(0, self.laser.pulse_duration, 1000)
        dt = t_arr[1] - t_arr[0]
        # Initialize result array with the same shape as k_y/k_z grid
        M_vals = np.zeros_like(self.ks[1], dtype=complex)
        
        # Process each k point in the grid
        for i in range(self.ks[1].shape[0]):
            for j in range(self.ks[1].shape[1]):
                k = np.array([self.ks[0][i,j], self.ks[1][i,j], self.ks[2][i,j]])
                k_norm = np.linalg.norm(k)
                
                # Define the integrand function for numerical integration
                def integrand(t):
                    A_t = self.laser.A(np.array([t]))
                    print(A_t)
                    A_t = A_t.reshape(3)
                    
                    exponent = 1j * ((k + A_t)**2 / 2 - 1j * self.ground_state_energy(k_norm)) * t
                    return np.dot(k, A_t) * np.exp(exponent)
                
                # Integrate over time
                result, _ = integrate.quad(integrand, 0, self.laser.pulse_duration, limit=1000, complex_func=True)
                print(result)
                
                M_vals[i, j] = -1j * self.ground_state_wf(k) * result 
        return M_vals
    
    def plot_dP_dk(self, ax):
        dP_dk = np.abs(self.M_SFA_VG) **2

        fig = ax.get_figure()
        fig.set_size_inches(8, 6)
        # Using k_y and k_z for plotting since k_x is all zeros
        contour = ax.contourf(self.ks[1], self.ks[2], dP_dk, levels=50, cmap='viridis')
        fig.colorbar(contour, ax=ax, label='dP/dk')
        ax.set_title('Differential Probability Distribution')
        ax.set_xlabel('k_y')
        ax.set_ylabel('k_z')