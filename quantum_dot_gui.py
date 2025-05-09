import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Constants
hbar = 1.055e-34  # Planck's constant [J*s]
m = 9.11e-31      # Electron mass [kg]
q = 1.6e-19       # Elementary charge [C]
pi = np.pi


def energy(n, L):
    """
    Return the nth eigenenergy of an electron in an infinite 1D potential well
    ("particle in a box") of width L.

    E_n = (n^2 π^2 ħ^2) / (2 m L^2)

    Parameters:
      n : int
          Quantum number (1,2,3,...)
      L : float
          Width of the box [meters]

    Returns:
      E_n : float
          Energy of level n [Joules]
    """
    
    #known solution
    return (n**2 * pi**2 * hbar**2) / (2 * m * L**2) 


def x_matrix_element(k, n, L):
    """
    Compute the electric-dipole matrix element <k| x |n> for the 1D box.

    Selection rule: only transitions with k+n odd are allowed (parity).
    The closed-form result is:

      ⟨k|x|n⟩ = (8 L / π^2) * (k*n) / ( (k^2 - n^2)^2 ),  if k+n is odd
                0                                         otherwise

    This integral quantifies how strongly an oscillating field can drive
    transitions between levels n and k.

    Parameters:
      k, n : int
          Quantum numbers of final and initial states.
      L : float
          Width of the box [meters].

    Returns:
      x_kn : float
          Dipole matrix element [meters].
    """
    
    # selection rule: only nonzero matrix element when k+n is odd
    if (k + n) % 2 == 1:
        return (8 * L / (pi**2)) * (k * n) / ((k**2 - n**2)**2)
    else:
        return 0


def emission_spectrum(n_initial, L, E0, n_max=10):
    """
    Build the emission spectrum from an excited state n_initial → all k < n_initial.

    For each lower state k, we compute:
      - Photon angular frequency ω = (E_n - E_k) / ħ
      - Transition intensity ∝ |⟨k|x|n_initial⟩|^2  (Fermi's Golden Rule)

    Parameters:
      n_initial : int
          Quantum number of the initially excited state.
      L : float
          Box width [m].
      E0 : float
          Electric field amplitude (arbitrary units).
      n_max : int
          (Unused) maximum level index — here we only go k = 1 … n_initial-1.

    Returns:
      spectrum : list of (ω, I) tuples
          ω   = photon angular frequency [rad/s]
          I   = relative transition rate (unitless)
    """
    spectrum = []
    # Energy of the excited state
    En = energy(n_initial, L)

    # Loop over all lower levels k = 1, 2, …, n_initial-1
    for k in range(1, n_initial):
        Ek = energy(k, L)                      # energy of level k
        omega = (En - Ek) / hbar               # transition frequency ω = ΔE/ħ
        x_kn = x_matrix_element(k, n_initial, L)  # dipole coupling
        # Fermi's Golden Rule: rate ∝ E0^2 * |x_kn|^2 / ħ
        intensity = (q**2 * E0**2 / hbar) * (x_kn**2)
        spectrum.append((omega, intensity))

    return spectrum  # list of (ω, I) pairs

def interactive_spectrum_plot():
    """
    Create a Matplotlib figure with:
      - A stem plot of the emission spectrum
      - A slider to vary the box width L (dot size)
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25) # make room at the bottom for the slider
    
    # Initial plot setup
    ax.set_title('Emission Spectrum for 1D Quantum Dot')
    ax.set_xlabel('Photon Energy ℏω (rad/s)')
    ax.set_ylabel('Transition Rate (arb. units)')
    ax.grid(True)

    # Default parameters
    initial_L = 10e-9  # start with a 10 nm dot
    E0 = 1e5           # field amplitude (arb. units)
    n_initial = 3      # excite to level n=3

    # Compute and plot the initial spectrum
    spectrum = emission_spectrum(n_initial, initial_L, E0)
    omega_vals, intensities = zip(*spectrum)
    line = ax.stem(omega_vals, intensities, basefmt=" ")
    markerline, stemlines, baseline = line



    # Slider controlling L from 5 nm to 20 nm
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Dot Size (nm)', 5, 20, valinit=10, valstep=0.5)

    # Callback to update the plot whenever the slider moves
    def update(val):
        L = slider.val * 1e-9 # convet nm to m
        spectrum = emission_spectrum(n_initial, L, E0)
        
        if spectrum:  # check that there is something to plot
            omega_vals, intensities = zip(*spectrum)
        else:
            omega_vals, intensities = [], []

        # Clear and redraw
        ax.cla()    
        ax.set_title(f'Emission Spectrum (L = {slider.val:.1f} nm)')
        ax.set_xlabel('Photon Energy ℏω (rad/s)')
        ax.set_ylabel('Transition Rate (arb. units)')
        ax.grid(True)
        ax.stem(omega_vals, intensities, basefmt=" ")
        fig.canvas.draw_idle()
        
    slider.on_changed(update)  # <-- Link the slider to the callback
    plt.show()                 # <-- Display the interactive plot

        
    

# run the GUI 
interactive_spectrum_plot()
