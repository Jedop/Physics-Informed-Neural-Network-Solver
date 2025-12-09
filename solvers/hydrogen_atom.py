# |------------Prerequisites------------|
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True to a scalar")

torch.set_grad_enabled(True)

def init_weights(m):
    if isinstance(m, nn.Linear):
        # The Paper's Secret Sauce: std = 0.01
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(m.bias)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

print(f"Using {device} device")

# |------------Model------------|
class PINN(nn.Module):
    def __init__(self, alpha_init=0.8, symmetry_break="none"):
        super().__init__()
        self.symmetry_break = symmetry_break
        self.linear_stack = nn.Sequential(
            nn.Linear(3, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )
        self.E = nn.Parameter(torch.tensor([0.0]))
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)]))

    def forward(self, x, y, z):
        inp = torch.cat([x, y, z], dim=1)
        logits_x = self.linear_stack(inp)
        r = torch.sqrt(x*x + y*y + z*z)
        decay = torch.exp(-torch.abs(self.alpha) * r)
        
        if self.symmetry_break == "z_axis":
            return logits_x * decay * z
        else:
            return logits_x * decay
    
    def loss_fn(self, x, y, z):
        psi_val = self.forward(x, y, z)

        psi_x = torch.autograd.grad(
            outputs=psi_val,
            inputs=x,
            grad_outputs=torch.ones_like(psi_val),
            create_graph=True,
            retain_graph=True
        )[0]
        psi_xx = torch.autograd.grad(
            outputs=psi_x,
            inputs=x,
            grad_outputs=torch.ones_like(psi_x),
            create_graph=True,
            retain_graph=True
        )[0]
        psi_y = torch.autograd.grad(
            outputs=psi_val,
            inputs=y,
            grad_outputs=torch.ones_like(psi_val),
            create_graph=True,
            retain_graph=True
        )[0]
        psi_yy = torch.autograd.grad(
            outputs=psi_y,
            inputs=y,
            grad_outputs=torch.ones_like(psi_y),
            create_graph=True,
            retain_graph=True
        )[0]
        psi_z = torch.autograd.grad(
            outputs=psi_val,
            inputs=z,
            grad_outputs=torch.ones_like(psi_val),
            create_graph=True,
            retain_graph=True
        )[0]
        psi_zz = torch.autograd.grad(
            outputs=psi_z,
            inputs=z,
            grad_outputs=torch.ones_like(psi_z),
            create_graph=True,
            retain_graph=True
        )[0]

        pde = (psi_xx + psi_yy + psi_zz + 2 * (self.E + 1/torch.sqrt(torch.square(x) + torch.square(y) + torch.square(z)))* psi_val)**2
        residual = torch.mean(pde)
        if self.symmetry_break == "none":
            normalization_loss = 1.0 / (torch.mean(psi_val ** 2) + 1e-6)
        else:
            normalization_loss = (torch.mean(psi_val ** 2) - 1.0) ** 2
        self.test = (
            "Residual Loss =", residual.detach().item(),
            "Normalization Loss =", normalization_loss.detach().item()
        )

        return residual + normalization_loss


# |------------Model and Optimizer Initialization------------|
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINN for Hydrogen Atom Quantum States")
    parser.add_argument("--n", type=int, choices=[1, 2, 4], default=1, 
                        help="Principal Quantum Number (1=Ground, 2=2p, 4=4d)")
    args = parser.parse_args()

    config = {
        1: {
            "alpha_init": 0.5,
            "energy_init": 0,
            "symmetry_break": "none", # S-orbital
            "log_interval": 500,
            "steps": 20000
        },
        2: {
            "alpha_init": 0.8,
            "energy_init": 0,
            "symmetry_break": "z_axis", # P-orbital (Dumbbell)
            "log_interval": 100, 
            "steps": 4000
        },
        4: {
            "alpha_init": 0.2,
            "energy_init": 0,
            "symmetry_break": "z_axis", # D-orbital (discovered state)
            "log_interval": 100,
            "steps": 4000
        }
    }
    n = args.n
    cfg = config[args.n]
    print(f"--- Initializing Hydrogen Atom Simulation for n={args.n} ---")
    print(f"Auto-configuring: Alpha={cfg['alpha_init']}, Mode={cfg['symmetry_break']}")

    # Initialize
    model = PINN(alpha_init=cfg["alpha_init"], symmetry_break=cfg["symmetry_break"])
    model.apply(init_weights)
    model.to(device)
    learning_rate = 2e-3
    epochs = cfg["steps"]
    optimizer = torch.optim.Adam(model.linear_stack.parameters(), lr=learning_rate)
    optimizer_E = torch.optim.Adam([model.E], lr=2e-3)
    scheduler_E = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_E, 'min', patience=100, factor=0.5)
    alpha_lr = 1e-2 if n != 1 else 1e-1
    optimizer_alpha = torch.optim.Adam([model.alpha], lr=alpha_lr)
    scheduler_alpha = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_alpha, 'min', patience=100, factor=0.5)

    # Space Sampling

    N = 500
    scale = 1.0  # controls concentration

    # Radial sampling (exponential)
    r = torch.distributions.Exponential(scale).sample((N, 1)) + 1e-4

    # Angular sampling (uniform sphere)
    phi = 2 * torch.pi * torch.rand((N, 1))
    cos_theta = 2 * torch.rand((N, 1)) - 1
    theta = torch.acos(cos_theta)

    # Spherical to Cartesian
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    x, y, z = x.to(device), y.to(device), z.to(device)
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True

    # |------------Model and Optimizer Initialization------------|
    lambdaphy = 1
    best_loss = float('inf')  # Initialize with infinity
    best_model_state = None
    for epoch in range(epochs):
        loss = lambdaphy * model.loss_fn(x=x, y=y, z=z)
        loss.backward()
        optimizer.step()
        optimizer_E.step()
        optimizer_alpha.step()
        scheduler_E.step(loss)
        scheduler_alpha.step(loss)
        optimizer.zero_grad()
        optimizer_alpha.zero_grad()
        optimizer_E.zero_grad()

        if loss.item() / lambdaphy < best_loss:
            best_loss = loss.item() / lambdaphy
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % cfg["log_interval"] == 0 and epoch != 0:
            lambdaphy += 1 if n != 1 else 0
            print(model.test, "Energy =", float(model.E), "alpha =", float(model.alpha))
            print(f"Epoch {epoch}: Loss = {loss.item()/lambdaphy}")


    print(f"Training finished. Best loss achieved: {best_loss}")
    model.load_state_dict(best_model_state)

    # |------------Evaluation------------|
    N = 2000
    scale = 1.0  # controls concentration

    # Radial sampling (exponential)
    r = torch.distributions.Exponential(scale).sample((N, 1)) + 1e-4

    # Angular sampling (uniform sphere)
    phi = 2 * torch.pi * torch.rand((N, 1))
    cos_theta = 2 * torch.rand((N, 1)) - 1
    theta = torch.acos(cos_theta)

    # Spherical to Cartesian
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    x, y, z = x.to(device), y.to(device), z.to(device)

    
    def plot_xz_heatmap(model):
        # 1. Create a grid
        grid_size = 100
        range_limit = 2 if n == 1 else 10 * (n - 1)
        
        x = torch.linspace(-range_limit, range_limit, grid_size)
        z = torch.linspace(-range_limit, range_limit, grid_size)
        X, Z = torch.meshgrid(x, z, indexing='ij')
        
        # 2. Setup Coordinates (Slice at y=0)
        x_flat = X.reshape(-1, 1)
        z_flat = Z.reshape(-1, 1)
        y_flat = torch.zeros_like(x_flat) # The slice plane
        
        x_flat, y_flat, z_flat = x_flat.to(device), y_flat.to(device), z_flat.to(device)

        # 3. Get Predictions
        model.eval()
        with torch.no_grad():
            psi_pred = model(x_flat, y_flat, z_flat).cpu().numpy()

        psi_pred = psi_pred / psi_pred.max()
            
        # 4. Reshape and Plot
        Psi_grid = psi_pred.reshape(grid_size, grid_size)

        plt.figure(figsize=(8, 8))
        # We plot Psi (not Psi^2) to see the Positive (Red) and Negative (Blue) lobes
        plt.contourf(X.numpy(), Z.numpy(), Psi_grid, levels=50, cmap='RdBu')
        plt.colorbar(label='Wavefunction Amplitude')
        plt.title(f"Hydrogen n={n} Orbital (xz-plane)\nE: {model.E.item():.4f}, Alpha: {model.alpha.item():.4f}")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.savefig(f"Hydrogen Atom n = {n} Heatmap.png")
        print("Figure Saved!")

    plot_xz_heatmap(model)

    def calculate_metrics(model, n=1):
        # 1. Sample a dense cloud of points to verify physics
        N_test = 10000
        r = torch.linspace(0.1, 15, N_test).view(-1, 1) # Avoid 0 singularity for analytical formula
        
        # We only test the Radial part because that's what the Energy determines
        # We assume x=r, y=0, z=0 (Line along x-axis)
        val = r / np.sqrt(2) 
        x = val
        y = torch.zeros_like(r)
        z = val 
        x, y, z = x.to(device), y.to(device), z.to(device)

        # 2. Get Prediction
        model.eval()
        with torch.no_grad():
            psi_pred = model(x, y, z).cpu().numpy().flatten()
        r = r.cpu().numpy().flatten()

        if n == 1: # 1s orbital
            psi_exact = np.exp(-r)
        elif n == 2: # 2p orbital (radial part ~ r * exp(-r/2))
            psi_exact = r * np.exp(-r/2)
        elif n == 4: 
            exit
        else:
            print("Unknown state for analytic comparison")
            return

        # We want to find scalar 'a' such that a * psi_pred approx psi_exact
        # Solution: a = dot(pred, exact) / dot(pred, pred)
        numerator = np.dot(psi_pred, psi_exact)
        denominator = np.dot(psi_pred, psi_pred)
        alpha = numerator / denominator
        
        psi_final = alpha * psi_pred

        # Calculate Error (Relative L2 Norm)
        # Formula: || exact - pred || / || exact ||
        diff = psi_exact - psi_final
        l2_error = np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(psi_exact**2))
        
        print(f"--- Accuracy Report (n={n}) ---")
        print(f"Optimal Scale Factor: {alpha:.4e}")
        print(f"Relative L2 Error:    {l2_error:.5f} ({l2_error*100:.3f}%)")
        
        return l2_error

    print(calculate_metrics(model, n=n)) 