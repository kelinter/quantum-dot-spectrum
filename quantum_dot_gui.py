import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Constants
hbar = 1.055e-34  # Planck's constant [J*s]
m = 9.11e-31      # Electron mass [kg]
q = 1.6e-19       # Elementary charge [C]
pi = np.pi

def energy(n, L):
    return (n**2 * pi**2 * hbar**2) / (2 * m * L**2)
