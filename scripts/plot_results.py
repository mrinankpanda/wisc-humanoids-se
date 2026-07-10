import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_results(npz_path, dt=0.02, save_path=None):
    data = np.load(npz_path)

    actual_pos = data["actual_body_pos"]
    est_pos = data["est_body_pos"]
    actual_vel = data["actual_body_vel"]
    est_vel = data["est_body_vel"]

    print("First 10 positions:")
    print(actual_pos[:10])

    print("\nMin position:")
    print(actual_pos.min(axis=0))

    print("Max position:")
    print(actual_pos.max(axis=0))

    n_steps = actual_pos.shape[0]
    t = np.arange(n_steps) * dt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axis_labels = ["x", "y", "z"]

    for i in range(3):
        ax = axes[0, i]
        ax.plot(t, actual_pos[:, i], label="ground truth", linewidth=2)
        ax.plot(t, est_pos[:, i], label="IEKF estimate", linestyle="--")
        ax.set_title(f"Position {axis_labels[i]}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("meters")
        ax.legend()

    for i in range(3):
        ax = axes[1, i]
        ax.plot(t, actual_vel[:, i], label="ground truth", linewidth=2)
        ax.plot(t, est_vel[:, i], label="IEKF estimate", linestyle="--")
        ax.set_title(f"Velocity {axis_labels[i]}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("m/s")
        ax.legend()

    plt.tight_layout()

    # --- simple error metrics ---
    pos_err = np.linalg.norm(actual_pos - est_pos, axis=1)
    vel_err = np.linalg.norm(actual_vel - est_vel, axis=1)

    print(f"Position error: mean={pos_err.mean():.4f} m, max={pos_err.max():.4f} m")
    print(f"Velocity error: mean={vel_err.mean():.4f} m/s, max={vel_err.max():.4f} m/s")

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", required=True, help="Path to results .npz file")
    parser.add_argument("--dt", type=float, default=0.02, help="Timestep used in simulation")
    parser.add_argument("--save-path", default=None, help="If set, saves plot instead of showing it")
    args = parser.parse_args()

    plot_results(args.npz_path, dt=args.dt, save_path=args.save_path)