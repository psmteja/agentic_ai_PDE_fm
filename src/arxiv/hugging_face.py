"""
pde_transformer_diffusion_finetune.py

Pipeline:
  1. Load pretrained PDE-Transformer (mc-s).
  2. Generate synthetic 2D heat-diffusion data with a finite-difference solver.
  3. Evaluate 1-step prediction:
        input  = [u(t0), u(t1)]
        target = u(t2)
     Compute:
        - MSE
        - Relative L2 error
        - "Accuracy" = max(0, 100 * (1 - RelL2))
  4. If dataset accuracy < 60%, fine-tune the model on diffusion data.
  5. Save fine-tuned weights to:
        - a .pth file (state_dict)
        - a Hugging Face–style folder (if model.save_pretrained exists)
  6. Re-evaluate and visualize before/after fine-tuning.
"""

import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
    Simple 2D Gaussian bump on [-1, 1]^2 with random center/width/amplitude.
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
# 2D heat / diffusion solver (periodic BCs)
# ================================================================

def simulate_diffusion_2d(u0: np.ndarray, nt: int, dt: float, nu: float) -> np.ndarray:
    """
    2D diffusion / heat equation with periodic boundaries:

      ∂_t u = ν (u_xx + u_yy)

    Returns u(t) with shape: (nt, nx, ny)
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


# ================================================================
# Dataset for diffusion
# ================================================================

class DiffusionDataset(Dataset):
    """
    Each sample is a short scalar-field sequence u(t) on a 2D grid, solved
    with the finite-difference diffusion solver.

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

        x_in = np.stack([u_t0, u_t1], axis=0).astype(np.float32)  # (2, nx, ny)
        y_target = u_t2.astype(np.float32)                        # (nx, ny)

        return (
            torch.from_numpy(x_in),
            torch.from_numpy(y_target),
            torch.from_numpy(u),
        )


def make_diffusion_generator(nx: int, ny: int, nt: int = 5) -> callable:
    """
    Returns a function that generates one diffusion trajectory u(t).
    """
    def generator():
        u0 = make_random_gaussian_bump(nx, ny)
        # moderately diffusive
        return simulate_diffusion_2d(u0, nt=nt, dt=0.05, nu=0.2)
    return generator


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
# Evaluation utilities (diffusion only)
# ================================================================

def evaluate_diffusion_dataset(
    model: PDETransformer,
    device,
    generator_fn,
    n_samples: int = 32,
    batch_size: int = 4,
):
    """
    Evaluate PDE-Transformer on diffusion dataset.

    Returns:
      mean_mse, mean_rel_l2, approx_accuracy(%)
    """
    dataset = DiffusionDataset(generator_fn, n_samples=n_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_mse = []
    all_rel_l2 = []

    model.eval()
    with torch.no_grad():
        for x_in, y_target, _u_full in dataloader:
            x_in = x_in.to(device, dtype=torch.float32)          # (B, 2, nx, ny)
            y_target = y_target.to(device, dtype=torch.float32)  # (B, nx, ny)

            model_out = model(x_in)
            y_pred_all = extract_prediction_tensor(model_out)    # (B, C, nx, ny) typically

            # Convention: channel 1 ~ "next state" prediction
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

    print("\n=== Dataset-level evaluation (diffusion_heat) ===")
    print(f"Dataset 1-step MSE:    mean={mean_mse:.6e}, std={std_mse:.6e}")
    print(f"Dataset 1-step Rel L2: mean={mean_rel_l2:.6e}, std={std_rel_l2:.6e}")
    print(f'Naive "accuracy" (1 - Rel L2): {approx_acc:.2f}%')

    return mean_mse, mean_rel_l2, approx_acc


def evaluate_single_example(
    model: PDETransformer,
    device,
    u_full: np.ndarray,
    title_prefix: str = "diffusion_heat",
):
    """
    Evaluate model on a single trajectory u_full (nt, nx, ny) and plot.
    """
    model.eval()

    u_t0 = u_full[0]
    gt_next = u_full[2]

    x_in = np.stack([u_full[0], u_full[1]], axis=0).astype(np.float32)  # (2, nx, ny)
    x_in_t = torch.from_numpy(x_in)[None, ...].to(device)               # (1, 2, nx, ny)

    with torch.no_grad():
        y_out = model(x_in_t)
    y_tensor = extract_prediction_tensor(y_out)  # (1, C, nx, ny)
    y_next = y_tensor[0, 1].detach().cpu().numpy()

    mse_single = np.mean((y_next - gt_next) ** 2)
    rel_l2_single = np.linalg.norm(y_next - gt_next) / (np.linalg.norm(gt_next) + 1e-12)
    single_acc = max(0.0, 100.0 * (1.0 - rel_l2_single))

    print(f"\n=== Single example evaluation ({title_prefix}) ===")
    print(f"Single example MSE (t2):    {mse_single:.6e}")
    print(f"Single example Rel L2:      {rel_l2_single:.6e}")
    print(f'Naive "accuracy" (single):  {single_acc:.2f}%')

    plot_single_step(f"{title_prefix} – single example", u_t0, gt_next, y_next)


# ================================================================
# Fine-tuning on diffusion
# ================================================================

def fine_tune_on_diffusion(
    model: PDETransformer,
    device,
    generator_fn,
    n_train_samples: int = 256,
    batch_size: int = 8,
    n_epochs: int = 5,
    lr: float = 4e-5,
):
    """
    Fine-tune PDE-Transformer on synthetic diffusion data using MSE loss
    between predicted u(t2) (channel 1) and finite-difference ground truth.
    """
    print("\n=== Starting fine-tuning on diffusion data ===")
    dataset = DiffusionDataset(generator_fn, n_samples=n_train_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        n_batches = 0

        for x_in, y_target, _u_full in dataloader:
            x_in = x_in.to(device, dtype=torch.float32)
            y_target = y_target.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            model_out = model(x_in)
            y_pred_all = extract_prediction_tensor(model_out)  # (B, C, nx, ny)
            y_pred = y_pred_all[:, 1]                          # (B, nx, ny)

            loss = loss_fn(y_pred, y_target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        print(f"Epoch {epoch}/{n_epochs} | Train MSE loss: {avg_loss:.6e}")

    print("=== Fine-tuning complete ===")


# ================================================================
# Saving helpers for Hugging Face
# ================================================================

def save_finetuned_model(model: PDETransformer, save_root: str = "pde_transformer_mc_s_diffusion_finetuned"):
    """
    Save:
      - A .pth file with the state_dict
      - A Hugging Face–style folder (if model.save_pretrained exists)
    """
    os.makedirs(save_root, exist_ok=True)

    # 1) Save plain PyTorch checkpoint (.pth)
    pth_path = os.path.join(save_root, "pde_transformer_mc_s_diffusion_finetuned.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"Saved state_dict checkpoint (.pth) to: {os.path.abspath(pth_path)}")

    # 2) Try to save in Hugging Face format
    #    This creates config.json + pytorch_model.bin (or safetensors).
    try:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(save_root)
            print(f"Saved Hugging Face format (config + weights) in: {os.path.abspath(save_root)}")
        else:
            # Minimal fallback: create pytorch_model.bin + tiny config.json
            hf_bin_path = os.path.join(save_root, "pytorch_model.bin")
            torch.save(model.state_dict(), hf_bin_path)

            config_path = os.path.join(save_root, "config.json")
            minimal_cfg = {
                "model_type": "pde_transformer_mc_s",
                "library_name": "pdetransformer",
                "base_model": "thuerey-group/pde-transformer",
                "subfolder": "mc-s",
                "description": "PDE-Transformer mc-s fine-tuned on 2D diffusion (heat equation).",
            }
            with open(config_path, "w") as f:
                json.dump(minimal_cfg, f, indent=2)

            print("Model does not have save_pretrained; wrote minimal HF files:")
            print(f"  - {os.path.abspath(hf_bin_path)}")
            print(f"  - {os.path.abspath(config_path)}")
    except Exception as e:
        print(f"Warning: failed to save in HF format: {e}")


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

    # 2D grid/time setup
    nx = ny = 64
    nt = 5
    diffusion_gen = make_diffusion_generator(nx, ny, nt)

    # Generate one fixed trajectory to use for before/after visualization
    u_full_test = diffusion_gen()  # (nt, nx, ny)

    # ------------------------------------------------------------
    # BEFORE fine-tuning
    # ------------------------------------------------------------
    print("\n=== BEFORE FINE-TUNING ===")
    _, _, acc_before = evaluate_diffusion_dataset(
        model, device, generator_fn=diffusion_gen,
        n_samples=32, batch_size=4,
    )
    evaluate_single_example(
        model, device, u_full=u_full_test,
        title_prefix="diffusion_heat – BEFORE fine-tuning",
    )

    # ------------------------------------------------------------
    # FINE-TUNE IF ACCURACY < 60%
    # ------------------------------------------------------------
    if acc_before < 60.0:
        print(f"\nAccuracy {acc_before:.2f}% < 60% → fine-tuning the model...")
        fine_tune_on_diffusion(
            model, device, generator_fn=diffusion_gen,
            n_train_samples=256, batch_size=8,
            n_epochs=5, lr=4e-5,
        )

        # Save fine-tuned weights (both .pth and HF-style folder)
        save_finetuned_model(model, save_root="pde_transformer_mc_s_diffusion_finetuned")
    else:
        print(f"\nAccuracy {acc_before:.2f}% ≥ 60% → skipping fine-tuning.")
        # Still save the (essentially) original model if you want
        save_finetuned_model(model, save_root="pde_transformer_mc_s_diffusion_evalonly")

    # ------------------------------------------------------------
    # AFTER (possible) fine-tuning
    # ------------------------------------------------------------
    print("\n=== AFTER (POSSIBLE) FINE-TUNING ===")
    _, _, acc_after = evaluate_diffusion_dataset(
        model, device, generator_fn=diffusion_gen,
        n_samples=32, batch_size=4,
    )
    evaluate_single_example(
        model, device, u_full=u_full_test,
        title_prefix="diffusion_heat – AFTER fine-tuning",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
