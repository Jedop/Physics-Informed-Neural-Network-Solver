# |------------Prerequisites------------|
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


class Sin(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)


torch.set_grad_enabled(True)


def ansatz(x, n):
    result = x * (1 - x)
    for k in range(1, n):
        result = result * (k / n - x)
    return result


# |------------Model------------|
class PINN(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.n = n
        layers = []
        layers.append(nn.Linear(1, n * 20))
        layers.append(nn.Tanh())

        # hidden layers
        for _ in range(n):
            layers.append(nn.Linear(n * 20, n * 20))
            layers.append(nn.Tanh())

        # final layer
        layers.append(nn.Linear(n * 20, 1))
        self.linear_stack = nn.Sequential(*layers)
        self.E = nn.Parameter(torch.tensor([(n - 1) ** 2 * torch.pi**2 / 2]))

    def forward(self, x):
        logits = self.linear_stack(x)
        psi_x = ansatz(x, self.n) * logits
        return psi_x

    def loss_fn(self, x):
        x = x.clone().detach().requires_grad_(True)
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
        residual = (
            1 / psi_val.shape[0] * torch.sum((psi_xx + 2 * self.E * psi_val) ** 2)
        )
        self.test = float(residual.detach())
        return 1 * (10 ** (self.n + 1)) * residual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve for the nth eigenstate of the Infinite Square Well using a PINN."
    )
    parser.add_argument(
        "n",
        type=int,
        help="The principal quantum number (n) of the eigenstate to find (e.g., 0, 1, 2 and 3).",
    )
    args = parser.parse_args()

    n_state = args.n
    print(f"Attempting to find the n={n_state} eigenstate.")
    # |------------Model and Optimizer Initialization------------|
    model = PINN(n=1)
    model.to(device)
    learning_rate = 1e-4
    epochs = 40001
    optimizer = torch.optim.Adam(model.linear_stack.parameters(), lr=learning_rate)
    optimizer_E = torch.optim.Adam([model.E], lr=5e-1)
    scheduler_E = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_E, "min", patience=500, factor=0.5
    )
    x = torch.linspace(0, 1, 1000).unsqueeze(1)
    x = x.to(device)
    x.requires_grad = True
    print(torch.Tensor.size(x))

    # |------------Model and Optimizer Initialization------------|
    lambdaphy = 1
    best_loss = float("inf")  # Initialize with infinity
    best_model_state = None
    E_values = []
    losses = []
    for epoch in range(epochs):
        loss = lambdaphy * model.loss_fn(x)
        loss.backward()
        optimizer.step()
        optimizer_E.step()
        scheduler_E.step(loss.item())
        optimizer.zero_grad()
        optimizer_E.zero_grad()
        if loss.item() / lambdaphy < best_loss:
            best_loss = loss.item() / lambdaphy
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            E_values.append(model.E.item())
            losses.append(loss.item() / lambdaphy)

        if epoch % 500 == 0:
            lambdaphy += 1.33 if epoch != 0 else 0
            # print(model.test, float(model.E.detach())) # Uncomment this line to debug
            print(f"Epoch {epoch}: Loss = {loss.item() / lambdaphy}")

    print(f"Training finished. Best loss achieved: {best_loss}")
    model.load_state_dict(best_model_state)

    # |------------Evaluation------------|

    model.eval()
    x = torch.linspace(0, 1, 1000).unsqueeze(1)
    x = x.to(device)
    with torch.no_grad():
        psi_pred = model(x).detach().cpu().numpy()
        E_learned = model.E.item()
    x = x.to("cpu")
    k = np.sqrt(2 * E_learned)
    psi_true = np.sin(k * x.numpy())
    psi_pred /= np.max(np.abs(psi_pred))
    psi_true /= np.max(np.abs(psi_true))
    Error = np.sum((psi_pred - psi_true) ** 2) / np.sum(psi_true**2)
    print(f"Error = {Error}")
    print(f"Percentage Error = {Error * 100:.4f}%")
    plt.figure(figsize=(7, 4))
    plt.plot(x, psi_pred, label="PINN Prediction", lw=2)
    plt.plot(x, psi_true, "--", label="Analytical Solution", lw=2)
    plt.title(
        f"ψ(x) — PINN vs Analytical (E = {E_learned:.4f}) with Error = {Error * 100:.4f}%"
    )
    plt.xlabel("x")
    plt.ylabel("ψ(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Infinite Square well n = {model.n}.png")
    plt.show()
