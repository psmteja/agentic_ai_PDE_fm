"""
evaluate_pde_transformer_usecases.py

Test PDE-Transformer (mc-s) on several toy 2D PDEs:
  1) Diffusion / heat equation
  2) Reaction–diffusion (Fisher–KPP-like)
  3) 2D Burgers (advection + diffusion, we evaluate only u_x)

For each case:
  - Generate sequences with a simple finite-difference solver
  - Use [u(t0), u(t1)] as model input
  - Compare model's predicted next state vs numerical solver's u(t2)
  - Report MSE, relative L2, and a naive "accuracy" = 1 - Rel L2
  - Plot one example

These PDE solvers are intentionally simple and not identical
to the training setup in the PDE-Transformer paper.
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from pdetransformer.core.mixed_channels import PDETransformer


# ================================================================
# General utilities
# ================================================================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_prediction_tensor(model_output) -> torch.Tensor:
    """
    Robust helper to get the actual prediction tensor out of PDETransformer.
    Handles:
      - raw Tensor
      - objects with `.prediction` or `.sample`
      - dict-like outputs
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    if hasattr(model_output, "prediction"):
        return model_output.prediction

    if hasattr(model_output, "sample"):
        return model_output.sample

    if isinstance(model_output, dict):
        for v in model_output.values():
            if isinstance(v, torch.Tensor):
                return v

    raise TypeError(
        f"Don't know how to get tensor from PDETransformer output. "
        f"type={type(model_output)}, dir={dir(model_output)}"
    )


def make_random_gaussian_bump(nx: int, ny: int) -> np.ndarray:
    """
    Simple 2D Gaussian bump on [-1, 1]^2 with random center/width.
    """
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    cx = np.random.uniform(-0.5, 0.5)
    cy = np.random.uniform(-0.5, 0.5)
    sigma = np.random.uniform(0.1, 0.4)
    amp = np.random.uniform(0.5, 1.0)

    u0 = amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2))
    return u0.astype(np.float32)


def make_random_smooth_field(nx: int, ny: int) -> np.ndarray:
    """
    Random smooth-ish field via diffused noise (for Burgers ICs).
    """
    noise = np.random.randn(nx, ny).astype(np.float32)
    # Diffuse a few steps to smooth it
    u = noise.copy()
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    dt = 0.1
    nu = 0.2
    for _ in range(10):
        u_xx = (np.roll(u, -1, axis=0) - 2.0 * u + np.roll(u, 1, axis=0)) / (dx ** 2)
        u_yy = (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / (dy ** 2)
        u = u + dt * nu * (u_xx + u_yy)
    u = u / np.max(np.abs(u) + 1e-8)
    return u.astype(np.float32)


# ================================================================
# Toy PDE solvers (all periodic BCs)
# ================================================================

def simulate_diffusion_2d(u0: np.ndarray, nt: int, dt: float, nu: float) -> np.ndarray:
    """
    2D diffusion / heat equation with periodic boundaries:

      ∂_t u = ν (u_xx + u_yy)
    """
    nx, ny = u0.shape
    u = np.zeros((nt, nx, ny), dtype=np.float32)
    u[0] = u0

    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    for t in range(1, nt):
        un = u[t - 1]
        u_xx = (np.roll(un, -1, axis=0) - 2.0 * un + np.roll(un, 1, axis=0)) / (dx ** 2)
        u_yy = (np.roll(un, -1, axis=1) - 2.0 * un + np.roll(un, 1, axis=1)) / (dy ** 2)
        u[t] = un + dt * nu * (u_xx + u_yy)

    return u


def simulate_fisher_kpp_2d(u0: np.ndarray, nt: int, dt: float, D: float, r: float) -> np.ndarray:
    """
    2D Fisher–KPP-like equation with periodic boundaries:

      ∂_t u = D (u_xx + u_yy) + r u (1 - u)

    Values are clamped to [0, 1] for stability.
    """
    nx, ny = u0.shape
    u = np.zeros((nt, nx, ny), dtype=np.float32)
    u[0] = np.clip(u0, 0.0, 1.0)

    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    for t in range(1, nt):
        un = u[t - 1]
        u_xx = (np.roll(un, -1, axis=0) - 2.0 * un + np.roll(un, 1, axis=0)) / (dx ** 2)
        u_yy = (np.roll(un, -1, axis=1) - 2.0 * un + np.roll(un, 1, axis=1)) / (dy ** 2)
        reaction = r * un * (1.0 - un)
        unew = un + dt * (D * (u_xx + u_yy) + reaction)
        u[t] = np.clip(unew, 0.0, 1.0)

    return u


def simulate_burgers_2d(u0x: np.ndarray, u0y: np.ndarray,
                         nt: int, dt: float, nu: float) -> np.ndarray:
    """
    2D Burgers-like system (toy advection + diffusion) with periodic BCs:

      ∂_t u + u u_x + v u_y = ν (u_xx + u_yy)
      ∂_t v + u v_x + v v_y = ν (v_xx + v_yy)

    Returns sequence of u_x only (shape: (nt, nx, ny)) so it matches
    the scalar-field training style in our other examples.
    """
    nx, ny = u0x.shape
    u = u0x.copy()
    v = u0y.copy()
    seq = np.zeros((nt, nx, ny), dtype=np.float32)
    seq[0] = u

    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    for t in range(1, nt):
        u_x = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dx)
        u_y = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dy)
        v_x = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dx)
        v_y = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dy)

        u_xx = (np.roll(u, -1, axis=0) - 2.0 * u + np.roll(u, 1, axis=0)) / (dx ** 2)
        u_yy = (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / (dy ** 2)
        v_xx = (np.roll(v, -1, axis=0) - 2.0 * v + np.roll(v, 1, axis=0)) / (dx ** 2)
        v_yy = (np.roll(v, -1, axis=1) - 2.0 * v + np.roll(v, 1, axis=1)) / (dy ** 2)

        u_adv = u * u_x + v * u_y
        v_adv = u * v_x + v * v_y

        u_new = u + dt * (-u_adv + nu * (u_xx + u_yy))
        v_new = v + dt * (-v_adv + nu * (v_xx + v_yy))

        # mild clipping to avoid blow-ups in long rollouts
        u = np.clip(u_new, -3.0, 3.0)
        v = np.clip(v_new, -3.0, 3.0)

        seq[t] = u

    return seq


# ================================================================
# Generic Dataset wrapper for different PDE cases
# ================================================================

class PDESingleFieldDataset(Dataset):
    """
    Each sample is a short scalar-field sequence u(t) on a 2D grid.

    - A generator function is passed that returns u(t) with shape (nt, nx, ny)
      for one randomly sampled initial condition.
    - Model input:  [u(t0), u(t1)]   (2, nx, ny)
    - Target:       u(t2)            (nx, ny)
    - Also returns the full sequence for visualization / rollouts.
    """

    def __init__(self, generator_fn, n_samples: int):
        super().__init__()
        self.generator_fn = generator_fn
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        u = self.generator_fn()  # (nt, nx, ny)
        u_t0 = u[0]
        u_t1 = u[1]
        u_t2 = u[2]

        x_in = np.stack([u_t0, u_t1], axis=0).astype(np.float32)   # (2, nx, ny)
        y_target = u_t2.astype(np.float32)                         # (nx, ny)

        return (
            torch.from_numpy(x_in),
            torch.from_numpy(y_target),
            torch.from_numpy(u),
        )


# ================================================================
# Visualization helpers
# ================================================================

def plot_single_step(title: str, u_t0, gt_next, pred_next):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)

    im0 = axes[0].imshow(u_t0, origin="lower")
    axes[0].set_title("u(t0) (ground truth)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(gt_next, origin="lower")
    axes[1].set_title("Ground truth u(t2)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(pred_next, origin="lower")
    axes[2].set_title("PDE-Transformer prediction")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.show()


# ================================================================
# Per-use-case generator functions
# ================================================================

def make_diffusion_generator(nx: int, ny: int, nt: int = 5) -> callable:
    def generator():
        u0 = make_random_gaussian_bump(nx, ny)
        # moderately diffusive
        return simulate_diffusion_2d(u0, nt=nt, dt=0.05, nu=0.2)
    return generator


def make_fisher_generator(nx: int, ny: int, nt: int = 5) -> callable:
    def generator():
        u0 = make_random_gaussian_bump(nx, ny)
        u0 = (u0 - u0.min()) / (u0.max() - u0.min() + 1e-8)  # normalize to [0, 1]
        D = np.random.uniform(0.001, 0.01)
        r = np.random.uniform(1.0, 5.0)
        return simulate_fisher_kpp_2d(u0, nt=nt, dt=0.02, D=D, r=r)
    return generator


def make_burgers_generator(nx: int, ny: int, nt: int = 5) -> callable:
    def generator():
        u0x = make_random_smooth_field(nx, ny)
        u0y = make_random_smooth_field(nx, ny)
        nu = np.random.uniform(0.01, 0.05)
        return simulate_burgers_2d(u0x, u0y, nt=nt, dt=0.01, nu=nu)
    return generator


# ================================================================
# Evaluation helper
# ================================================================

def evaluate_use_case(
    case_name: str,
    generator_fn,
    model: PDETransformer,
    device,
    n_samples: int = 32,
    batch_size: int = 4,
):
    print(f"\n{'=' * 80}")
    print(f"Evaluating use case: {case_name}")
    print(f"{'=' * 80}")

    dataset = PDESingleFieldDataset(generator_fn, n_samples=n_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_mse = []
    all_rel_l2 = []

    with torch.no_grad():
        for x_in, y_target, _u_full in dataloader:
            x_in = x_in.to(device, dtype=torch.float32)       # (B, 2, nx, ny)
            y_target = y_target.to(device, dtype=torch.float32)  # (B, nx, ny)

            model_out = model(x_in)
            y_pred_all = extract_prediction_tensor(model_out)    # (B, C, nx, ny) typically

            # Convention (following earlier examples): channel 1 ~ "next state" prediction
            y_pred = y_pred_all[:, 1]                            # (B, nx, ny)

            diff = y_pred - y_target

            mse_batch = torch.mean(diff ** 2, dim=(1, 2))        # (B,)
            all_mse.append(mse_batch.cpu().numpy())

            l2_diff = torch.sqrt(torch.sum(diff ** 2, dim=(1, 2)))
            l2_true = torch.sqrt(torch.sum(y_target ** 2, dim=(1, 2))) + 1e-12
            rel_l2_batch = (l2_diff / l2_true).cpu().numpy()
            all_rel_l2.append(rel_l2_batch)

    all_mse = np.concatenate(all_mse, axis=0)
    all_rel_l2 = np.concatenate(all_rel_l2, axis=0)

    mean_mse = all_mse.mean()
    std_mse = all_mse.std()
    mean_rel_l2 = all_rel_l2.mean()
    std_rel_l2 = all_rel_l2.std()
    approx_acc = max(0.0, 100.0 * (1.0 - mean_rel_l2))

    print(f"Dataset 1-step MSE:    mean={mean_mse:.6e}, std={std_mse:.6e}")
    print(f"Dataset 1-step Rel L2: mean={mean_rel_l2:.6e}, std={std_rel_l2:.6e}")
    print(f'Naive "accuracy" (1 - Rel L2): {approx_acc:.2f}%')

    # Visualize a single example from this use case
    x_in_single, y_target_single, u_full = dataset[0]
    u_full_np = u_full.numpy()
    u_t0 = u_full_np[0]
    gt_next = u_full_np[2]

    x_in_single = x_in_single.unsqueeze(0).to(device)  # (1, 2, nx, ny)
    with torch.no_grad():
        y_out = model(x_in_single)
    y_tensor = extract_prediction_tensor(y_out)
    y_next = y_tensor[0, 1].detach().cpu().numpy()

    mse_single = np.mean((y_next - gt_next) ** 2)
    rel_l2_single = np.linalg.norm(y_next - gt_next) / (np.linalg.norm(gt_next) + 1e-12)
    single_acc = max(0.0, 100.0 * (1.0 - rel_l2_single))

    print(f"\nSingle example MSE:      {mse_single:.6e}")
    print(f"Single example Rel L2:   {rel_l2_single:.6e}")
    print(f'Naive "accuracy" (single): {single_acc:.2f}%')

    plot_single_step(f"{case_name} – single example", u_t0, gt_next, y_next)


# ================================================================
# Main
# ================================================================

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load pretrained PDE-Transformer (mixed-channel, small)
    print("Loading PDE-Transformer mc-s ...")
    model = PDETransformer.from_pretrained(
        "thuerey-group/pde-transformer",
        subfolder="mc-s",
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded PDE-Transformer mc-s with {n_params:,} parameters")

    # Grid / time setup (same across use cases for simplicity)
    nx = ny = 64
    nt = 5
    n_samples = 32
    batch_size = 4

    # 2) Define use cases and associated generators
    use_cases = [
        ("diffusion_heat",  make_diffusion_generator(nx, ny, nt)),
        ("fisher_reaction_diffusion", make_fisher_generator(nx, ny, nt)),
        ("burgers_advection_diffusion", make_burgers_generator(nx, ny, nt)),
    ]

    # 3) Evaluate each
    for case_name, gen_fn in use_cases:
        evaluate_use_case(
            case_name=case_name,
            generator_fn=gen_fn,
            model=model,
            device=device,
            n_samples=n_samples,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    main()
