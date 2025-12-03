"""
diffusion_pde_transformer_finetune.py

Workflow:
  - Use PDE-Transformer (mc-s) as a base model.
  - Test it on a 2D heat / diffusion equation:
        ∂_t u = ν (u_xx + u_yy)
  - Compare 1-step prediction u(t2) vs ground truth from a finite-difference solver.
  - Compute:
        * MSE
        * relative L2 error
        * "accuracy" = 100 * (1 - Rel L2)
  - If dataset accuracy < 60%, fine-tune the model on diffusion data.
  - Re-evaluate and re-predict on the *same* test example before/after fine-tuning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from pdetransformer.core.mixed_channels import PDETransformer


# ================================================================
# Utilities
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


# ================================================================
# 2D diffusion / heat solver (periodic BCs)
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


def make_diffusion_generator(nx: int, ny: int, nt: int = 5) -> callable:
    """
    Returns a function that generates one random diffusion trajectory
    u(t) with shape (nt, nx, ny).
    """
    def generator():
        u0 = make_random_gaussian_bump(nx, ny)
        return simulate_diffusion_2d(u0, nt=nt, dt=0.05, nu=0.2)
    return generator


# ================================================================
# Dataset
# ================================================================

class DiffusionDataset(Dataset):
    """
    Each sample is a short heat-equation sequence:

      - We generate a random Gaussian IC u0
      - Solve forward with simulate_diffusion_2d to get u(t0..)
      - Model input:  [u(t0), u(t1)]  -> (2, nx, ny)
      - Target:       u(t2)           -> (nx, ny)
      - Also returns the whole sequence for visualization / rollouts.
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

        x_in = np.stack([u_t0, u_t1], axis=0).astype(np.float32)  # (2, nx, ny)
        y_target = u_t2.astype(np.float32)                        # (nx, ny)

        return (
            torch.from_numpy(x_in),
            torch.from_numpy(y_target),
            torch.from_numpy(u),
        )


# ================================================================
# Visualization
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
# Evaluation on a diffusion dataset
# ================================================================

def evaluate_diffusion_dataset(
    model: PDETransformer,
    device: torch.device,
    generator_fn,
    n_samples: int = 32,
    batch_size: int = 4,
):
    """
    Evaluate 1-step prediction performance for diffusion.

    Returns:
      mean_mse, mean_rel_l2, approx_accuracy (in %)
    """
    dataset = DiffusionDataset(generator_fn, n_samples=n_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_mse = []
    all_rel_l2 = []

    model.eval()
    with torch.no_grad():
        for x_in, y_target, _u_full in dataloader:
            x_in = x_in.to(device, dtype=torch.float32)           # (B, 2, nx, ny)
            y_target = y_target.to(device, dtype=torch.float32)   # (B, nx, ny)

            model_out = model(x_in)
            y_pred_all = extract_prediction_tensor(model_out)     # (B, C, nx, ny) typically

            # By convention: channel 1 = next state prediction
            y_pred = y_pred_all[:, 1]                             # (B, nx, ny)

            diff = y_pred - y_target

            mse_batch = torch.mean(diff ** 2, dim=(1, 2))         # (B,)
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

    print("\n" + "=" * 80)
    print("Diffusion dataset evaluation")
    print("=" * 80)
    print(f"Dataset 1-step MSE:    mean={mean_mse:.6e}, std={std_mse:.6e}")
    print(f"Dataset 1-step Rel L2: mean={mean_rel_l2:.6e}, std={std_rel_l2:.6e}")
    print(f'Naive "accuracy" (1 - Rel L2): {approx_acc:.2f}%')

    return mean_mse, mean_rel_l2, approx_acc


# ================================================================
# Single example: compare prediction vs ground truth
# ================================================================

def evaluate_single_example(
    model: PDETransformer,
    device: torch.device,
    u_full: np.ndarray,
    title_prefix: str = "diffusion_heat",
):
    """
    u_full: (nt, nx, ny) ground truth sequence from the PDE solver.

    Uses frames t0 and t1 as input, compares predicted u(t2) with ground truth.
    """
    model.eval()

    nt, nx, ny = u_full.shape
    u_t0 = u_full[0]
    gt_next = u_full[2]

    x_in = np.stack([u_full[0], u_full[1]], axis=0).astype(np.float32)  # (2, nx, ny)
    x_in_t = torch.from_numpy(x_in)[None, ...].to(device)               # (1, 2, nx, ny)

    with torch.no_grad():
        y_out = model(x_in_t)
    y_tensor = extract_prediction_tensor(y_out)                         # (1, C, nx, ny)
    y_next = y_tensor[0, 1].detach().cpu().numpy()

    mse_single = np.mean((y_next - gt_next) ** 2)
    rel_l2_single = np.linalg.norm(y_next - gt_next) / (np.linalg.norm(gt_next) + 1e-12)
    single_acc = max(0.0, 100.0 * (1.0 - rel_l2_single))

    print(f"\n{title_prefix} – single example results")
    print(f"Single example MSE:    {mse_single:.6e}")
    print(f"Single example Rel L2: {rel_l2_single:.6e}")
    print(f'Single example "accuracy": {single_acc:.2f}%')

    plot_single_step(f"{title_prefix} – single example", u_t0, gt_next, y_next)


# ================================================================
# Fine-tuning on diffusion data
# ================================================================

def fine_tune_on_diffusion(
    model: PDETransformer,
    device: torch.device,
    generator_fn,
    n_train_samples: int = 256,
    batch_size: int = 8,
    n_epochs: int = 5,
    lr: float = 4e-5,
):
    """
    Fine-tune PDE-Transformer on synthetic diffusion data using supervised MSE
    between predicted u(t2) and ground-truth u(t2).
    """
    print("\n" + "=" * 80)
    print("Starting fine-tuning on diffusion data...")
    print("=" * 80)

    train_dataset = DiffusionDataset(generator_fn, n_train_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        n_batches = 0

        for x_in, y_target, _u_full in train_loader:
            x_in = x_in.to(device, dtype=torch.float32)          # (B, 2, nx, ny)
            y_target = y_target.to(device, dtype=torch.float32)  # (B, nx, ny)

            optimizer.zero_grad()

            out = model(x_in)
            y_all = extract_prediction_tensor(out)               # (B, C, nx, ny)
            y_pred = y_all[:, 1]                                 # (B, nx, ny)

            loss = loss_fn(y_pred, y_target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        print(f"Epoch {epoch:03d}/{n_epochs} - train MSE loss: {avg_loss:.6e}")

    print("Fine-tuning finished.\n")
    model.eval()


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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded PDE-Transformer mc-s with {n_params:,} parameters")

    # Grid/time setup
    nx = ny = 64
    nt = 5

    # Diffusion generator
    diffusion_gen = make_diffusion_generator(nx, ny, nt)

    # 2) Create a fixed test trajectory for before/after comparison
    u_full_test = diffusion_gen()  # (nt, nx, ny)

    print("\n=== BEFORE FINE-TUNING ===")
    # Evaluate dataset performance
    _, _, acc_before = evaluate_diffusion_dataset(
        model,
        device,
        generator_fn=diffusion_gen,
        n_samples=32,
        batch_size=4,
    )

    # Single example prediction (same u_full_test will be used after fine-tuning too)
    evaluate_single_example(
        model,
        device,
        u_full=u_full_test,
        title_prefix="diffusion_heat – BEFORE fine-tuning",
    )

    # 3) If accuracy < 60%, fine-tune on diffusion data
    if acc_before < 60.0:
        print(f"\nAccuracy {acc_before:.2f}% < 60% → fine-tuning the model...")
        fine_tune_on_diffusion(
            model,
            device,
            generator_fn=diffusion_gen,
            n_train_samples=256,
            batch_size=8,
            n_epochs=5,
            lr=4e-5,
        )
    else:
        print(f"\nAccuracy {acc_before:.2f}% ≥ 60% → skipping fine-tuning.")

    print("\n=== AFTER (POSSIBLE) FINE-TUNING ===")
    # Re-evaluate on dataset
    _, _, acc_after = evaluate_diffusion_dataset(
        model,
        device,
        generator_fn=diffusion_gen,
        n_samples=32,
        batch_size=4,
    )

    # Re-run prediction on the SAME test trajectory as before
    evaluate_single_example(
        model,
        device,
        u_full=u_full_test,
        title_prefix="diffusion_heat – AFTER fine-tuning",
    )

    print(f"\nFinal dataset accuracy: {acc_after:.2f}%")

if __name__ == "__main__":
    main()
