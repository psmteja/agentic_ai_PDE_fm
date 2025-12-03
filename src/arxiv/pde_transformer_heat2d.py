import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from pdetransformer.core.mixed_channels import PDETransformer


# -----------------------------------------------------------------------------
# Utilities: IC generation, PDE solver, model output extraction
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_random_gaussian_ic(nx: int, ny: int) -> np.ndarray:
    """
    Create a random 2D Gaussian bump on [-1, 1]^2.
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


def generate_heat_2d(u0: np.ndarray, nt: int = 5, dt: float = 0.05, nu: float = 0.2) -> np.ndarray:
    """
    Simple 2D heat equation integrator with periodic BCs.

    ∂_t u = ν (∂_xx u + ∂_yy u)
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
        lap = u_xx + u_yy

        u[t] = un + dt * nu * lap

    return u


def extract_prediction_tensor(model_output) -> torch.Tensor:
    """
    Robust helper to get the actual prediction tensor out of PDETransformer.
    Handles:
    - raw Tensor
    - objects with .prediction
    - dict-like outputs
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    if hasattr(model_output, "prediction"):
        return model_output.prediction

    if isinstance(model_output, dict):
        # Take the first value
        return next(iter(model_output.values()))

    raise TypeError(
        f"Don't know how to get tensor from PDETransformer output. "
        f"type={type(model_output)}, dir={dir(model_output)}"
    )


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class HeatSequenceDataset(Dataset):
    """
    Each sample is a short heat-equation sequence:
      - We generate a random Gaussian IC u0
      - Solve forward to get u(t0), u(t1), u(t2), ...
      - Input to model: [u(t0), u(t1)]  (2 channels)
      - Target: u(t2)  (single field)
      - We also return the whole sequence for visualization / multi-step rollouts
    """

    def __init__(
        self,
        n_samples: int = 64,
        nx: int = 64,
        ny: int = 64,
        nt: int = 5,
        dt: float = 0.05,
        nu: float = 0.2,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.dt = dt
        self.nu = nu

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx):
        u0 = make_random_gaussian_ic(self.nx, self.ny)
        u = generate_heat_2d(u0, nt=self.nt, dt=self.dt, nu=self.nu)  # (nt, nx, ny)

        u_t0 = u[0]
        u_t1 = u[1]
        u_t2 = u[2]

        # Model input: (2, nx, ny)
        x_in = np.stack([u_t0, u_t1], axis=0).astype(np.float32)
        # Target: u(t2) -> (nx, ny)
        y_target = u_t2.astype(np.float32)

        # Return the full trajectory as well, for visualization / rollout
        return (
            torch.from_numpy(x_in),
            torch.from_numpy(y_target),
            torch.from_numpy(u),
        )


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def plot_single_step(u_t0, gt_next, pred_next):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axes[0].imshow(u_t0, origin="lower")
    axes[0].set_title("u(t0)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(gt_next, origin="lower")
    axes[1].set_title("Ground truth u(t2)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(pred_next, origin="lower")
    axes[2].set_title("PDE-Transformer pred")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.show()


def plot_rollout_comparison(u_true_seq, u_pred_seq, times_to_show=None):
    """
    u_true_seq, u_pred_seq: (nt, nx, ny) numpy arrays
    """
    nt = u_true_seq.shape[0]
    if times_to_show is None:
        times_to_show = [0, nt // 2, nt - 1]

    ncols = len(times_to_show)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))

    for i, t in enumerate(times_to_show):
        im_true = axes[0, i].imshow(u_true_seq[t], origin="lower")
        axes[0, i].set_title(f"True t={t}")
        plt.colorbar(im_true, ax=axes[0, i], shrink=0.8)

        im_pred = axes[1, i].imshow(u_pred_seq[t], origin="lower")
        axes[1, i].set_title(f"Pred t={t}")
        plt.colorbar(im_pred, ax=axes[1, i], shrink=0.8)

    axes[0, 0].set_ylabel("Ground truth")
    axes[1, 0].set_ylabel("Model rollout")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load pretrained PDETransformer
    model = PDETransformer.from_pretrained(
        "thuerey-group/pde-transformer",
        subfolder="mc-s",
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded PDE-Transformer mc-s with {n_params:,} parameters")

    # 2. Create dataset & dataloader
    nx = ny = 64
    nt = 5
    dataset = HeatSequenceDataset(
        n_samples=64,
        nx=nx,
        ny=ny,
        nt=nt,
        dt=0.05,
        nu=0.2,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 3. Evaluate 1-step MSE over the dataset
    all_mse = []

    with torch.no_grad():
        for x_in, y_target, _u_full in dataloader:
            # x_in: (B, 2, nx, ny)
            # y_target: (B, nx, ny)
            x_in = x_in.to(device, dtype=torch.float32)
            y_target = y_target.to(device, dtype=torch.float32)

            model_out = model(x_in)
            y_pred_all = extract_prediction_tensor(model_out)  # (B, C, nx, ny) typically

            # Assume channel 1 is the "next state" prediction
            y_pred = y_pred_all[:, 1]  # (B, nx, ny)

            mse_batch = torch.mean((y_pred - y_target) ** 2, dim=(1, 2))  # (B,)
            all_mse.append(mse_batch.cpu().numpy())

    all_mse = np.concatenate(all_mse, axis=0)
    print(f"Dataset 1-step MSE: mean={all_mse.mean():.6e}, std={all_mse.std():.6e}")

    # 4. Visualize a single example
    x_in, y_target, u_full = dataset[0]
    u_full = u_full.numpy()  # (nt, nx, ny)
    u_t0 = u_full[0]
    gt_next = u_full[2]

    x_in_single = x_in.unsqueeze(0).to(device)  # (1, 2, nx, ny)

    with torch.no_grad():
        y_out = model(x_in_single)
    y_tensor = extract_prediction_tensor(y_out)  # (1, C, nx, ny)
    y_next = y_tensor[0, 1].detach().cpu().numpy()

    mse_single = np.mean((y_next - gt_next) ** 2)
    rel_l2_single = np.linalg.norm(y_next - gt_next) / np.linalg.norm(gt_next)
    print(f"Single example MSE (t2): {mse_single:.6e}, Rel L2: {rel_l2_single:.6e}")

    plot_single_step(u_t0, gt_next, y_next)

    # 5. Autoregressive rollout starting from t0, t1
    print("\nRunning autoregressive rollout...")
    with torch.no_grad():
        states_model = [u_full[0], u_full[1]]  # two initial states
        nt_roll = u_full.shape[0]

        for t in range(2, nt_roll):
            # Build input from last two model states
            x_roll = np.stack([states_model[-2], states_model[-1]], axis=0).astype(np.float32)
            x_roll_t = torch.from_numpy(x_roll)[None, ...].to(device)  # (1, 2, nx, ny)

            y_out_t = model(x_roll_t)
            y_tensor_t = extract_prediction_tensor(y_out_t)
            next_state = y_tensor_t[0, 1].detach().cpu().numpy()  # (nx, ny)

            states_model.append(next_state)

    u_model_roll = np.stack(states_model, axis=0)  # (nt_roll, nx, ny)
    rollout_mse_per_t = np.mean((u_model_roll - u_full) ** 2, axis=(1, 2))

    print("Rollout MSE per time step:")
    for t, mse_t in enumerate(rollout_mse_per_t):
        print(f"  t={t}: MSE={mse_t:.6e}")

    plot_rollout_comparison(u_full, u_model_roll, times_to_show=[0, 2, nt_roll - 1])


if __name__ == "__main__":
    main()
