# Physics-Informed Neural Networks for Quantum Systems

**Status:** Actively in Progress

## Project Overview

This project is an implementation of Physics-Informed Neural Networks (PINNs) in PyTorch to solve the 1D time-independent Schrödinger equation for classic quantum mechanical systems. Unlike traditional neural networks that rely purely on data, a PINN's loss function is augmented with a term that enforces the validity of the underlying physical laws (in this case, the Schrödinger equation), allowing it to solve complex differential equations. In this project, the concept is taken further, and only Physical laws are used.

## Key Features

-   **3D Cartesian Solver:**: Successfully solves the Hydrogen atom without reducing it to a 1D radial problem, capturing angular dependencies and symmetries.
-   **Versatile PINN Application:** The core PINN architecture and physics-informed loss function were successfully adapted to solve for the eigenstates in multiple distinct quantum systems (Infinite Square Well and Harmonic Oscillator).
-   **Trainable Energy Levels:** The model treats the energy `E` as a trainable parameter, allowing it to discover the quantized energy eigenvalues of a system.
-   **High-Accuracy Solutions:** Achieves **< 0.2% error** for the ground and excited state wavefunctions and energy eigenvalues for implemented systems. For the ground state energy level of the infinite square well, it achieves a remarkable 2.4 x $10^{-7}$% error.

## Implemented Systems & Results

### 1. The Hydrogen Atom

The model solves the full **3-Dimensional Time-Independent Schrödinger Equation** in Cartesian coordinates $(x, y, z)$. The radial simplification is NOT used in this model, instead it solves the entire 3D equation, in cartesian coordinates.

**Key Achievements:**
 *   **High Precision:** Achieved a Relative L2 Error of **0.008%** for the Ground State and **0.012%** for the First Excited State compared to analytical solutions.
 *   **Spectral Targeting:** By adjusting the initialization of the learnable decay parameter $\alpha$, the network was guided to converge to specific energy levels ($n=2, n=4$).
 *   **Symmetry Breaking:** The model successfully captured non-spherical symmetries, spontaneously generating the **dumbbell shape ($p$-orbital)** and **multi-lobed structures ($d$-orbital)** to satisfy orthogonality constraints.

| Feature | **Ground State ($1s$)** <br> $E \approx -0.5$ | **First Excited State ($2p_z$)** <br> $E \approx -0.125$ | **High Energy State ($4d_{xz}$)** <br> $E \approx -0.033$ |
| :---: | :---: | :---: | :---: |
| **Volumetric Cloud** <br> *(Monte Carlo)* | ![1s Cloud](plots/Hydrogen%20Atom%20n%20=%201%20Cloud.png) | ![2p Cloud](plots/Hydrogen%20Atom%20n%20=%202%20Cloud.png) | ![4d Cloud](plots/Hydrogen%20Atom%20n%20=%204%20Cloud.png) |
| **3D Isosurface** <br> *(Plotly)* | ![1s Iso](plots/Hydrogen%20Atom%20n%20=%201%203D%20Modified.png) | ![2p Iso](plots/Hydrogen%20Atom%20n%20=%202%203D%20Modified.png) | ![4d Iso](plots/Hydrogen%20Atom%20n%20=%204%203D%20Modified.png) |
| **2D Cross-Section** <br> *(Heatmap)* | ![1s Heatmap](plots/Hydrogen%20Atom%20n%20=%201%20Heatmap.png) | ![2p Heatmap](plots/Hydrogen%20Atom%20n%20=%202%20Heatmap.png) | ![4d Heatmap](plots/Hydrogen%20Atom%20n%20=%204%20Heatmap.png) |

### 2. The Infinite Square Well

The model successfully learns the quantized energy levels and wavefunctions for the first three principal quantum numbers.

|n = 1|n = 2|n = 3|
|-----|-----|-----|
|![Infinite Square Well n=1](plots/Infinite%20Square%20well%20n%20=%201.png)|![Infinite Square Well n=2](plots/Infinite%20Square%20well%20n%20=%202.png)|![Infinite Square Well n=3](plots/Infinite%20Square%20well%20n%20=%203.png)|

### 3. The Quantum Harmonic Oscillator (SHO)

The model correctly discovers the evenly-spaced energy spectrum and the Gaussian-based wavefunctions for the ground and first excited states.

|n = 0|n = 1|
|-----|-----|
|![Infinite Square Well n=1](plots/Harmonic%20oscillator%20n%20=%200.png)|![Infinite Square Well n=2](plots/Harmonic%20oscillator%20n%20=%201.png)|

## Methodology and Challenges

### The Architecture: Physics-Informed Ansatz
Instead of using a "Black Box" neural network, this project employs a **Parametric Ansatz** approach. The wavefunction is modeled as:

$$ \psi(x) = \text{NN}(\mathbf{r}) \times A(\mathbf{r}) $$

Where $\text{NN}(\mathbf{r})$ is the neural network output, $A(\mathbf{r})$ is a function that enforces boundary conditions, and $\mathbf{r}$ is the position vector in cartesian Coordinates.

The total loss function is a combination of two components:

1.  **Physics Loss (`L_residual`):** This is the core of the PINN. It evaluates the Schrödinger equation `(Hψ - Eψ)` at various points in the domain. The loss is the mean squared error of this expression, driving it towards zero and thus forcing the network's output `ψ` to be a valid solution.
2.  **Normalization Loss (`L_normalization`):** This loss enforces the physical requirement that the wavefunction be normalizable. It penalizes the model if the integral of the probability density $`|ψ|^2`$ over the domain deviates from 1 (i.e., `<ψ|ψ> = 1`).

$$ Total Loss = λ_phy * (L_residual + L_normalization) $$

### The solution to the Hydrogen Singularity

The Potential in the Hydrogen Atom is inversely proportional to $r$, thus introducing a $-1/r$ term in the loss function which blows up at the origin $(r \to 0)$. This causes standard PINN approaches to explode or converge to trivial equations ($\psi = 0$).

1.  **The Pure MLP method:** I initially tried a standard MLP using Cartesian Coordinates, which failed to capture the radial symmetry, and also tended to go to 0 at the origin to avoid blowing up. This method seems to be an open problem in research, so I moved on(for now).
2.  **The Solution (Learnable decay):** I decided to introduce the boundary conditions $(\psi \neq 0$ at $r \= 0)$ and $(\psi \to 0 as r \to \infty)$ as a trial solution instead of loss terms. This is exactly what I have done for the other systems as well. The ansatz was: $\psi(r) \propto \text{NN}(x, y, z) \cdot e^{-|\alpha|r}$. This acts as a soft boundary condition towards $\infty$ and a hard boundary condition at 0, forcing the model to find a wavefunction that is non-zero at $r=0$.
3.  **Spectral Targetting:** This approach yielded an emergent feature: Spectral Targeting via Initialization. By initializing $\alpha$ to different values(provided the appropriate ansatz), the model is biased towards specific local minima, thus allowing us to effectively target higher order excited states that would otherwise be lost in the optimizing process. E.g. the results for n = 4 is actually obtained by simply changing the initialization of alpha in the n = 2 case from ~0.8 to ~0.2.

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

- The energy levels for the infinite square well start at 1, whereas for the harmonic oscillator it starts at 0. This is due to a difference in the mathematical formulations, and to stay true to the physics, I have decided to go with the conventions established instead of standardizing the energy levels.
- The Hydrogen Atom Solver is not built for the n = 3 case. To do so, you would have to update it with the appropriate boundary conditions, and tune the hyperparameters accordingly.
## Tech Stack

-   **Core Framework:** PyTorch
-   **Mathematics:** NumPy
-   **Visualization:** Matplotlib (2D), Plotly (3D Volumetric Rendering)

---
