# Physics-Informed Neural Networks for 1D Quantum Systems

**Status:** Actively in Progress

## Project Overview

This project is an implementation of Physics-Informed Neural Networks (PINNs) in PyTorch to solve the 1D time-independent Schrödinger equation for classic quantum mechanical systems. Unlike traditional neural networks that rely purely on data, a PINN's loss function is augmented with a term that enforces the validity of the underlying physical laws (in this case, the Schrödinger equation), allowing it to solve complex differential equations. In this project, the concept is taken further, and only Physical laws are used.

## Key Features

-   **Versatile PINN Application:** The core PINN architecture and physics-informed loss function were successfully adapted to solve for the eigenstates in multiple distinct quantum systems (Infinite Square Well and Harmonic Oscillator).
-   **Trainable Energy Levels:** The model treats the energy `E` as a trainable parameter, allowing it to discover the quantized energy eigenvalues of a system.
-   **High-Accuracy Solutions:** Achieves **< 0.2% error** for the ground and excited state wavefunctions and energy eigenvalues for implemented systems. For the ground state energy level of the infinite square well, it achieves a remarkable 2.4 x $10^{-7}$% error.

## Implemented Systems & Results

### 1. The Infinite Square Well

The model successfully learns the quantized energy levels and wavefunctions for the first three principal quantum numbers.

|n = 1|n = 2|n = 3|
|-----|-----|-----|
|![Infinite Square Well n=1](plots/Infinite%20Square%20well%20n%20=%201.png)|![Infinite Square Well n=2](plots/Infinite%20Square%20well%20n%20=%202.png)|![Infinite Square Well n=3](plots/Infinite%20Square%20well%20n%20=%203.png)|


### 2. The Quantum Harmonic Oscillator (SHO)

The model correctly discovers the evenly-spaced energy spectrum and the Gaussian-based wavefunctions for the ground and first excited states.

|n = 0|n = 1|
|-----|-----|
|![Infinite Square Well n=1](plots/Harmonic%20oscillator%20n%20=%200.png)|![Infinite Square Well n=2](plots/Harmonic%20oscillator%20n%20=%201.png)|

### 3. (In Progress) The Hydrogen Atom
Work is currently underway to extend this framework to solve the radial Schrödinger equation for the Hydrogen atom.

## How It Works

The total loss function is a combination of two components:

1.  **Physics Loss (`L_residual`):** This is the core of the PINN. It evaluates the Schrödinger equation `(Hψ - Eψ)` at various points in the domain. The loss is the mean squared error of this expression, driving it towards zero and thus forcing the network's output `ψ` to be a valid solution.
2.  **Normalization Loss (`L_normalization`):** This loss enforces the physical requirement that the wavefunction be normalizable. It penalizes the model if the integral of the probability density $`|ψ|^2`$ over the domain deviates from 1 (i.e., `<ψ|ψ> = 1`).

`Total Loss = λ_phy * (L_residual + L_normalization)`

---

The boundary conditions are enforced in the forward pass, by using a sort of trial solution based on the nodes of the given eigenstate.

For example,
1. Energy level n = 1 of the infinite square well => nodes are at 0, and a. Trial solution is x(a - x)
2. Energy level n = 2 of the infinite square well => nodes are at 0, a, and a/2. Trial solutiion is x(a-x)(a/2 - x)
3. Energy level n = 1 of the harmonic oscillator => nodes are at -inf, +inf, and 0. Given that the input space is finite, let L be the boundary of the input space. Then, trial solution is x(x^2 - L^2)

## Installation and Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/Jedop/Physics-Informed-Neural-Network-Solver.git
   cd Physics-Informed-Neural-Network-Solver
   ```
2. Install the required dependencies
   ```sh
   pip install -r requirements.txt
   ```
3. Run the script at the required energy level:
   E.g.
   ```sh
    # Example for the Harmonic Oscillator ground state (n=0)
    python harmonic_oscillator.py --n 0

    # Example for the Infinite Square Well first excited state (n=2)
    python infinite_square_well.py --n 2
   ```

## Points to Note

The energy levels for the infinite square well start at 1, whereas for the harmonic oscillator it starts at 0. This is due to a difference in the mathematical formulations, and to stay true to the physics, I have decided to go with the conventions established instead of standardizing the energy levels.

## Tech Stack

-   Python
-   PyTorch
-   NumPy
-   Matplotlib

---
