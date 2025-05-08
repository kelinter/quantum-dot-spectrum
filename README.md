# Quantum Dot Emission Spectrum Simulator

This Python project models a **1D quantum dot** as a **particle in a box**, and simulates its **emission spectrum** under the influence of a **time-dependent electric field**.

The simulation uses:
- **Time-dependent perturbation theory**
- **Fermi’s Golden Rule**
- **Dipole transition matrix elements**
- A real-time **interactive GUI** to vary the size of the quantum dot

---

## 🔬 Background

As the size of a quantum dot changes, the spacing between its energy levels also changes. This affects the frequency of emitted photons during electronic transitions, causing a **shift in the optical emission spectrum**. This simulation demonstrates that concept in real time.

---

## 📈 Features

- Models emission from a single excited quantum state
- Varies quantum dot size using a slider (from 5–20 nm)
- Displays allowed transitions based on matrix elements
- Interactive Matplotlib GUI

---

## ▶️ How to Run

1. Make sure you have Python 3 and pip installed.
2. Install required libraries:

```bash
pip install matplotlib numpy
