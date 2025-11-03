# |------------Prerequisites------------|
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.special import factorial, hermite

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


class Sin(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)


def harmonic_oscillator_wavefunction(n, x):
    """Return normalized wavefunction ψ_n(x) for 1D harmonic oscillator."""
    Hn = hermite(n)
    prefactor = 1.0 / np.sqrt((2**n) * factorial(n) * np.sqrt(np.pi))
    return prefactor * Hn(x) * np.exp(-(x**2) / 2)


torch.set_grad_enabled(True)


# |------------Model------------|
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 200),
            Sin(),
            nn.Linear(200, 200),
            Sin(),
            nn.Linear(200, 200),
            Sin(),
            nn.Linear(200, 200),
            Sin(),
            nn.Linear(200, 200),
            Sin(),
            nn.Linear(200, 1),
        )
        self.E = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        logits = self.linear_stack(x)
        psi_x = logits
        L = -float(x[0])

        return x * (x * x - L**2) * psi_x

    def loss_fn(self, x):
        psi_val = self.forward(x)
        psi_x = torch.autograd.grad(
            outputs=psi_val,
            inputs=x,
            grad_outputs=torch.ones_like(psi_val),
            create_graph=True,
            retain_graph=True,
        )[0]
        psi_xx = torch.autograd.grad(
            outputs=psi_x,
            inputs=x,
            grad_outputs=torch.ones_like(psi_x),
            create_graph=True,
            retain_graph=True,
        )[0]
        psi_at_boundaries = torch.cat((psi_val[0], psi_val[-1]))
        boundary_loss = torch.mean(psi_at_boundaries**2)
        residual = (
            1
            / psi_val.shape[0]
            * torch.sum((psi_xx + 2 * (self.E - x * x / 2) * psi_val) ** 2)
        )
        integral_psi_squared = torch.trapezoid(y=psi_val.squeeze() ** 2, x=x.squeeze())
        normalization_loss = (integral_psi_squared - 1.0) ** 2
        self.test = (float(residual), float(boundary_loss), float(normalization_loss))
        return residual + normalization_loss


# |------------Model and Optimizer Initialization------------|
model = PINN()
model.to(device)
learning_rate = 1e-5
epochs = 40001
optimizer = torch.optim.Adam(model.linear_stack.parameters(), lr=learning_rate)
optimizer_E = torch.optim.Adam([model.E], lr=5e-3)
scheduler_E = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_E, "min", patience=100, factor=0.5
)
x = torch.linspace(-8, 8, 4000).unsqueeze(1)
x = x.to(device)
x.requires_grad = True

# |------------Model and Optimizer Initialization------------|
lambdaphy = 1
best_loss = float("inf")  # Initialize with infinity
best_model_state = None
for epoch in range(epochs):
    loss = lambdaphy * model.loss_fn(x)
    loss.backward()
    optimizer.step()
    optimizer_E.step()
    scheduler_E.step(loss)
    optimizer.zero_grad()
    optimizer_E.zero_grad()

    if loss.item() / lambdaphy < best_loss:
        best_loss = loss.item() / lambdaphy
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 100 == 0 and epoch != 0:
        lambdaphy += 5
        print(model.test, float(model.E))
        print(f"Epoch {epoch}: Loss = {loss.item() / lambdaphy}")


print(f"Training finished. Best loss achieved: {best_loss}")
model.load_state_dict(best_model_state)

# |------------Evaluation------------|

model.eval()
x = torch.linspace(-8, 8, 4000).unsqueeze(1)
x = x.to(device)
with torch.no_grad():
    psi_pred = model(x).detach().cpu().numpy()
    E_learned = model.E.item()

# analytical solution
x_np = x.detach().cpu().numpy().flatten()
psi_true = -harmonic_oscillator_wavefunction(1, x_np)
# normalize both for fair comparison
psi_pred /= np.max(np.abs(psi_pred))
psi_true /= np.max(np.abs(psi_true))
psi_pred = psi_pred.squeeze()
x = x.to("cpu")
Error = np.sum((psi_pred - psi_true) ** 2) / np.sum(psi_true**2)
print(f"Error = {Error}")
print(f"Percentage Error = {Error * 100:.4f}%")
plt.figure(figsize=(7, 4))
plt.plot(x, psi_pred, label="PINN Prediction", lw=2)
plt.plot(x, psi_true, "--", label="Analytical Solution", lw=2)
plt.title(f"ψ(x) — PINN vs Analytical (E = {E_learned:.4f}) with Error = {Error}")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.legend()
plt.grid(True)
plt.savefig("Latest run for harmonic oscillator n = 2")
plt.show()
