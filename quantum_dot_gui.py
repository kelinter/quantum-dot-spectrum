import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

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


def absorption_spectrum(n_initial, L, E0, n_max=10):
    """
    Absorption: transitions from level n_initial up to k > n_initial.
    Returns list of (ω, intensity).
    """
    spec = []
    En = energy(n_initial, L)
    for k in range(n_initial+1, n_max+1):
        Ek = energy(k, L)
        ω = (Ek - En) / hbar
        xkn = x_matrix_element(k, n_initial, L)
        I = (q**2 * E0**2 / hbar) * (xkn**2)
        spec.append((ω, I))
    return spec



def interactive_spectrum_plot():
    """
    Create a Matplotlib figure with:
      - A stem plot of the emission spectrum
      - A slider to vary the box width L (dot size)
    """
    
    # Set up figure + axes
    fig, ax = plt.subplots(figsize=(8,5))
    # leave space on the right for the legend
    plt.subplots_adjust(left=0.3, right=0.75, bottom=0.25)
    
    # Initial parameters
    mode        = 'Emission'  # start in Emission mode
    dot_size_nm = 10          # initial L in nm
    E0          = 1e5         # field amplitude (arb.)
    n_emit      = 3           # for emission: start in level 3
    n_absorb    = 1           # for absorption: start in ground
    n_max       = 10          # max level to consider
    
    def plot_spectrum():
        ax.clear()
        L = dot_size_nm * 1e-9
        ax.set_title(f'{mode} Spectrum (L = {dot_size_nm:.1f} nm)')
        ax.set_xlabel('Photon Energy ℏω (rad/s)')
        ax.set_ylabel('Transition Rate (arb. units)')
        ax.grid(True)

        if mode == 'Emission':
            spec = emission_spectrum(n_emit, L, E0)
        else:
            spec = absorption_spectrum(n_absorb, L, E0, n_max)

        if spec:
            ω_vals, I_vals = zip(*spec)
            ax.stem(ω_vals, I_vals, basefmt=" ")
            
        # place the legend outside the plot on the left
        fig.legend(loc='lower left',
                  bbox_to_anchor=(1.02, 0.5), title="Mode",
                  frameon=True)


        fig.canvas.draw_idle()

    # Initial draw
    plot_spectrum()

    # Slider for dot size
    ax_slider = plt.axes([0.3, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Dot Size (nm)', 5, 20, valinit=dot_size_nm, valstep=0.5)
    def on_slider(val):
        nonlocal dot_size_nm
        dot_size_nm = slider.val
        plot_spectrum()
    slider.on_changed(on_slider)


    # Radio buttons for Emission/Absorption
    ax_radio = plt.axes([0.05, 0.5, 0.15, 0.15], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(ax_radio, ('Emission', 'Absorption'), active=0)
    def on_radio(label):
        nonlocal mode
        mode = label
        plot_spectrum()
    radio.on_clicked(on_radio)

    plt.show()                 # <-- Display the interactive plot

        
    

# run the GUI 
interactive_spectrum_plot()
