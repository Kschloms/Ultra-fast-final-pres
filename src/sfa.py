import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        # Gaussian envelope function for the laser pulse
        envelope = np.zeros_like(t)
        envelope[(t >= 0) & (t <= self.pulse_duration)] = (np.sin(np.pi * t[(t >= 0) & (t <= self.pulse_duration)] / self.pulse_duration)) ** 2
        return envelope
    
    def plot_envelope(self, t: np.ndarray) -> None:
        envelope = self._envelope(t)
        plt.figure(figsize=(10, 6))
        plt.plot(t, envelope, label='Envelope', color='blue')
        plt.title(f"Envelope of {self.name} Laser Pulse")
        plt.xlabel("Time (fs)")
        plt.ylabel("Amplitude (a.u.)")
        plt.legend()
        plt.grid()
        plt.show()

    def A(self, t, phase: float = np.pi) -> np.ndarray:
        A_0 = np.sqrt(self.intensity_au) / self.frequency_au
        A_x = np.zeros_like(t)  # x-component is zero for linearly polarized light
        A_y = A_0 * self._envelope(t) * self.epsilon * np.sin(self.frequency_au * t + phase) / np.sqrt(1 - self.epsilon**2)
        A_z = A_0 * self._envelope(t) * np.cos(self.frequency_au * t + phase) / np.sqrt(1 - self.epsilon**2)
        return np.array([A_x, A_y, A_z])

    def plot_electric_field(self, phase: float = 0.0) -> None:
        t = np.linspace(-self.pulse_duration, self.pulse_duration, 1000)
        A = self.A(t, phase)
        plt.figure(figsize=(10, 6))
        plt.plot(t, A[1], label='Electric Field (y-component)', color='blue')
        plt.plot(t, A[2], label='Electric Field (z-component)', color='red')
        plt.title(f"Electric Field of {self.name} Laser Pulse")
        plt.xlabel("Time (fs)")
        plt.ylabel("Electric Field (a.u.)")
        plt.legend()
        plt.grid()
        plt.show()

    

class Photo_Electron:
    def __init__(self, energy: float, momentum: float):
        self.energy = energy  # in eV
        self.momentum = momentum  # in kg*m/s

    def __str__(self):
        return f"Photoelectron: {self.energy} eV, {self.momentum} kg*m/s"