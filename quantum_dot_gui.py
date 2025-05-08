import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Constants
hbar = 1.055e-34  # Planck's constant [J*s]
m = 9.11e-31      # Electron mass [kg]
q = 1.6e-19       # Elementary charge [C]
pi = np.pi

def energy(n, L):
    #known solution
    return (n**2 * pi**2 * hbar**2) / (2 * m * L**2) 

def x_matrix_element(k, n, L):
    # selection rule: only nonzero matrix element when k+n is odd
    if (k + n) % 2 == 1:
        return (8 * L / (pi**2)) * (k * n) / ((k**2 - n**2)**2)
    else:
        return 0

def emission_spectrum(n_initial, L, E0, n_max=10):
    spectrum = []
    for k in range(1, n_initial):
        En = energy(n_initial, L)
        Ek = energy(k, L)
        omega = (En - Ek) / hbar
        xnk = x_matrix_element(k, n_initial, L)
        intensity = (q**2 * E0**2 / hbar) * (xnk**2)
        spectrum.append((omega, intensity))
    return spectrum

def interactive_spectrum_plot():
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.set_title('Emission Spectrum for 1D Quantum Dot')
    ax.set_xlabel('Photon Energy ℏω (rad/s)')
    ax.set_ylabel('Transition Rate (arb. units)')
    ax.grid(True)

    initial_L = 10e-9  # 10 nm
    E0 = 1e5
    n_initial = 3

    spectrum = emission_spectrum(n_initial, initial_L, E0)
    omega_vals, intensities = zip(*spectrum)
    line = ax.stem(omega_vals, intensities, basefmt=" ")


    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Dot Size (nm)', 5, 20, valinit=10, valstep=0.5)

    def update(val):
        L = slider.val * 1e-9
        spectrum = emission_spectrum(n_initial, L, E0)
        omega_vals, intensities = zip(*spectrum)
        ax.cla()
        ax.set_title(f'Emission Spectrum (L = {slider.val:.1f} nm)')
        ax.set_xlabel('Photon Energy ℏω (rad/s)')
        ax.set_ylabel('Transition Rate (arb. units)')
        ax.grid(True)
        ax.stem(omega_vals, intensities, basefmt=" ", use_line_collection=True)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    
interactive_spectrum_plot()
