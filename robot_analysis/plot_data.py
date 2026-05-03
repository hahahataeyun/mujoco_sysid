from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("/home/capture15/shared_data/capture/test_pd/ik/6/raw/arm")
OUTPUT_DIR = DATA_DIR / "plots"

PLOT_FILES = {
    "action_qpos": "action_qpos.npy",
    "position": "position.npy",
    "torque": "torque.npy",
    "velocity": "velocity.npy",
}


def load_array(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return np.asarray(data.tolist() if data.dtype == object else data, dtype=float)


def load_time(data_len: int) -> tuple[np.ndarray, str]:
    time_path = DATA_DIR / "time.npy"
    if not time_path.exists():
        return np.arange(data_len), "sample"

    time = load_array(time_path).reshape(-1)
    if len(time) != data_len:
        return np.arange(data_len), "sample"

    return time - time[0], "time [s]"


def plot_joint_data(name: str, filename: str) -> None:
    data = load_array(DATA_DIR / filename)
    if data.ndim != 2:
        raise ValueError(f"{filename} must be a 2D array, got shape {data.shape}")

    x, xlabel = load_time(len(data))
    fig, ax = plt.subplots(figsize=(12, 6))

    for joint_idx in range(data.shape[1]):
        ax.plot(x, data[:, joint_idx], label=f"joint {joint_idx + 1}", linewidth=1.2)

    ax.set_title(name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(name)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_action_qpos_vs_position() -> None:
    action_qpos = load_array(DATA_DIR / "action_qpos.npy")
    position = load_array(DATA_DIR / "position.npy")
    if action_qpos.shape != position.shape:
        raise ValueError(
            "action_qpos.npy and position.npy must have the same shape, "
            f"got {action_qpos.shape} and {position.shape}"
        )
    if action_qpos.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got shape {action_qpos.shape}")

    x, xlabel = load_time(len(action_qpos))
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for joint_idx in range(action_qpos.shape[1]):
        color = colors[joint_idx % len(colors)]
        ax.plot(
            x,
            action_qpos[:, joint_idx],
            label=f"joint {joint_idx + 1} action_qpos",
            color=color,
            linestyle="-",
            linewidth=1.2,
        )
        ax.plot(
            x,
            position[:, joint_idx],
            label=f"joint {joint_idx + 1} position",
            color=color,
            linestyle="--",
            linewidth=1.2,
        )

    ax.set_title("action_qpos vs position")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("qpos")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "action_qpos_vs_position.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_action_qpos_minus_position() -> None:
    action_qpos = load_array(DATA_DIR / "action_qpos.npy")
    position = load_array(DATA_DIR / "position.npy")
    if action_qpos.shape != position.shape:
        raise ValueError(
            "action_qpos.npy and position.npy must have the same shape, "
            f"got {action_qpos.shape} and {position.shape}"
        )
    if action_qpos.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got shape {action_qpos.shape}")

    diff = action_qpos - position
    x, xlabel = load_time(len(diff))
    marker_interval = max(len(diff) // 200, 1)
    selected_joint_indices = [0, 2, 5]

    for joint_idx in selected_joint_indices:
        if joint_idx >= diff.shape[1]:
            raise ValueError(
                f"Requested joint {joint_idx + 1}, but data only has {diff.shape[1]} joints"
            )

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            x,
            diff[:, joint_idx],
            label=f"joint {joint_idx + 1} diff",
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            markevery=marker_interval,
        )
        ax.plot(
            x,
            np.sign(diff[:, joint_idx]),
            label=f"joint {joint_idx + 1} sign",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"joint {joint_idx + 1}: action_qpos - position")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("qpos error / sign")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        output_path = OUTPUT_DIR / f"action_qpos_minus_position_joint{joint_idx + 1}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Saved {output_path}")


def plot_torque_zoom_1s() -> None:
    torque = load_array(DATA_DIR / "torque.npy")
    if torque.ndim != 2:
        raise ValueError(f"torque.npy must be a 2D array, got shape {torque.shape}")

    x, xlabel = load_time(len(torque))
    start_time = x[len(x) // 2]
    end_time = start_time + 1.0
    mask = (x >= start_time) & (x <= end_time)
    if mask.sum() < 2:
        start_idx = len(torque) // 2
        end_idx = min(start_idx + 100, len(torque))
        mask = np.zeros(len(torque), dtype=bool)
        mask[start_idx:end_idx] = True

    fig, ax = plt.subplots(figsize=(12, 6))
    for joint_idx in range(torque.shape[1]):
        ax.plot(
            x[mask],
            torque[mask, joint_idx],
            label=f"joint {joint_idx + 1}",
            linewidth=1.2,
            marker="o",
            markersize=2.5,
        )

    ax.set_title(f"torque zoom: {x[mask][0]:.3f}s to {x[mask][-1]:.3f}s")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("torque")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "torque_zoom_1s.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in PLOT_FILES.items():
        plot_joint_data(name, filename)
    plot_torque_zoom_1s()
    plot_action_qpos_vs_position()
    plot_action_qpos_minus_position()


if __name__ == "__main__":
    main()
