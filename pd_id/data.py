from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ARM_DOF = 6


@dataclass(frozen=True)
class ArmDataset:
    action_qpos: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray
    time: np.ndarray
    dt: float

    @property
    def length(self) -> int:
        return int(self.position.shape[0])


def _load_float_array(path: Path, ndim: int | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path, allow_pickle=True)
    array = np.asarray(array, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{path} must have ndim={ndim}, got shape={array.shape}")
    return array


def _as_arm_series(path: Path) -> np.ndarray:
    array = _load_float_array(path, ndim=2)
    if array.shape[1] != ARM_DOF:
        raise ValueError(f"{path} must have {ARM_DOF} columns, got shape={array.shape}")
    return array


def _median_dt(time: np.ndarray) -> float:
    if len(time) < 2:
        raise ValueError("time.npy needs at least two samples")
    diffs = np.diff(time)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if len(diffs) == 0:
        raise ValueError("time.npy does not contain increasing timestamps")
    return float(np.median(diffs))


def load_arm_dataset(data_dir: Path) -> ArmDataset:
    action_qpos = _as_arm_series(data_dir / "action_qpos.npy")
    position = _as_arm_series(data_dir / "position.npy")
    velocity = _as_arm_series(data_dir / "velocity.npy")
    torque = _as_arm_series(data_dir / "torque.npy")
    time = _load_float_array(data_dir / "time.npy", ndim=1)

    n = min(len(action_qpos), len(position), len(velocity), len(torque), len(time))
    if n < 3:
        raise ValueError(f"not enough samples in {data_dir}")

    action_qpos = action_qpos[:n]
    position = position[:n]
    velocity = velocity[:n]
    torque = torque[:n]
    time = time[:n]

    finite = (
        np.all(np.isfinite(action_qpos), axis=1)
        & np.all(np.isfinite(position), axis=1)
        & np.all(np.isfinite(velocity), axis=1)
        & np.all(np.isfinite(torque), axis=1)
        & np.isfinite(time)
    )
    if not np.all(finite):
        action_qpos = action_qpos[finite]
        position = position[finite]
        velocity = velocity[finite]
        torque = torque[finite]
        time = time[finite]

    order = np.argsort(time)
    action_qpos = action_qpos[order]
    position = position[order]
    velocity = velocity[order]
    torque = torque[order]
    time = time[order]

    return ArmDataset(
        action_qpos=action_qpos,
        position=position,
        velocity=velocity,
        torque=torque,
        time=time,
        dt=_median_dt(time),
    )


@dataclass(frozen=True)
class WindowBatch:
    qpos0: np.ndarray
    qvel0: np.ndarray
    controls: np.ndarray
    target_qpos: np.ndarray
    target_qvel: np.ndarray
    target_tau: np.ndarray
    mask: np.ndarray


def sample_window_batch(
    dataset: ArmDataset,
    *,
    batch_size: int,
    min_steps: int,
    max_steps: int,
    rng: np.random.Generator,
) -> WindowBatch:
    if min_steps < 1 or max_steps < min_steps:
        raise ValueError(f"invalid window steps: min={min_steps}, max={max_steps}")
    if dataset.length <= max_steps + 1:
        raise ValueError(
            f"dataset has {dataset.length} samples, but max_steps={max_steps} requires "
            f"at least {max_steps + 2}"
        )

    starts = rng.integers(0, dataset.length - max_steps - 1, size=batch_size)
    lengths = rng.integers(min_steps, max_steps + 1, size=batch_size)
    offsets = np.arange(max_steps)
    src = starts[:, None] + offsets[None, :]
    target = src + 1
    mask = offsets[None, :] < lengths[:, None]

    return WindowBatch(
        qpos0=dataset.position[starts],
        qvel0=dataset.velocity[starts],
        controls=dataset.action_qpos[src],
        target_qpos=dataset.position[target],
        target_qvel=dataset.velocity[target],
        target_tau=dataset.torque[target],
        mask=mask.astype(np.float64),
    )


def sample_fixed_window_batch(
    dataset: ArmDataset,
    *,
    batch_size: int,
    steps: int,
    rng: np.random.Generator,
) -> WindowBatch:
    return sample_window_batch(
        dataset,
        batch_size=batch_size,
        min_steps=steps,
        max_steps=steps,
        rng=rng,
    )


def full_sequence_batch(dataset: ArmDataset) -> WindowBatch:
    return WindowBatch(
        qpos0=dataset.position[:1],
        qvel0=dataset.velocity[:1],
        controls=dataset.action_qpos[:-1][None, ...],
        target_qpos=dataset.position[1:][None, ...],
        target_qvel=dataset.velocity[1:][None, ...],
        target_tau=dataset.torque[1:][None, ...],
        mask=np.ones((1, dataset.length - 1), dtype=np.float64),
    )
