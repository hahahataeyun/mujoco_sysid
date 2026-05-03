from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_PATH = Path("/home/capture15/shared_data/capture/eccv2026/inspire_f1")
JOINT_NAMES = ("little", "ring", "middle", "index", "thumb_2", "thumb_1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display right hand commands against right joint states per joint."
    )
    parser.add_argument(
        "object_name",
        help="Object sequence name under the capture dataset, e.g. apple.",
    )
    parser.add_argument(
        "episode",
        help="Episode number under the object sequence, e.g. 0.",
    )
    return parser.parse_args()


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)


def load_hand_series(hand_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    commands = load_npy(hand_dir / "right_commands.npy")
    states = load_npy(hand_dir / "right_joint_states.npy")
    command_times = load_npy(hand_dir / "right_commands_time.npy")
    state_times = load_npy(hand_dir / "right_joint_states_time.npy")

    if commands.ndim != 2 or states.ndim != 2:
        raise ValueError(
            "Expected right_commands.npy and right_joint_states.npy to be 2D arrays, "
            f"got {commands.shape} and {states.shape}."
        )
    if commands.shape[1] != states.shape[1]:
        raise ValueError(
            "Command/state joint counts differ: "
            f"{commands.shape[1]} vs {states.shape[1]}."
        )
    if command_times.shape[0] != commands.shape[0]:
        raise ValueError(
            "right_commands_time.npy length does not match right_commands.npy: "
            f"{command_times.shape[0]} vs {commands.shape[0]}."
        )
    if state_times.shape[0] != states.shape[0]:
        raise ValueError(
            "right_joint_states_time.npy length does not match right_joint_states.npy: "
            f"{state_times.shape[0]} vs {states.shape[0]}."
        )

    return commands, states, command_times, state_times


def plot_commands_vs_states(
    object_name: str,
    episode: str,
    commands: np.ndarray,
    states: np.ndarray,
    command_times: np.ndarray,
    state_times: np.ndarray,
) -> None:
    joint_count = commands.shape[1]
    time_zero = min(float(command_times[0]), float(state_times[0]))
    command_t = command_times - time_zero
    state_t = state_times - time_zero

    fig, axes = plt.subplots(joint_count, 1, sharex=True, figsize=(12, 2.2 * joint_count))
    axes = np.atleast_1d(axes)
    fig.suptitle(f"{object_name}/{episode}: right_commands vs right_joint_states")

    for joint_idx, axis in enumerate(axes):
        joint_name = JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"joint_{joint_idx}"
        axis.plot(command_t, commands[:, joint_idx], label="right_commands", linewidth=1.4)
        axis.plot(state_t, states[:, joint_idx], label="right_joint_states", linewidth=1.2)
        axis.set_ylabel(joint_name)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Time since first sample (s)")
    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    hand_dir = BASE_PATH / args.object_name / str(args.episode) / "raw" / "hand"
    commands, states, command_times, state_times = load_hand_series(hand_dir)
    plot_commands_vs_states(
        args.object_name,
        str(args.episode),
        commands,
        states,
        command_times,
        state_times,
    )


if __name__ == "__main__":
    main()
