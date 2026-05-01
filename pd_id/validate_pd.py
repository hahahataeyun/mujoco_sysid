from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from pd_id.data import load_arm_dataset
from pd_id.model import arm_layout, load_mujoco_model, require_mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/home/capture15/shared_data/capture/test_pd/cartesian/2/raw/arm")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pd_id_results/validation"
DEFAULT_FITTED_DIR = REPO_ROOT / "pd_id_results"


@dataclass(frozen=True)
class Segment:
    start: int
    steps: int

    @property
    def stop(self) -> int:
        return self.start + self.steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a fitted xArm MJCF on held-out recorded PD data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--robot-xml", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w-pos", type=float, default=100.0)
    parser.add_argument("--w-vel", type=float, default=10.0)
    parser.add_argument("--w-tau", type=float, default=1.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def _steps_from_seconds(seconds: float, dt: float) -> int:
    return max(1, int(round(seconds / dt)))


def _select_segment(dataset_length: int, steps: int, start: int | None, seed: int) -> Segment:
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if dataset_length <= steps + 1:
        raise ValueError(
            f"dataset has {dataset_length} samples, but validation needs at least {steps + 2}"
        )
    if start is None:
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, dataset_length - steps - 1))
    if start < 0 or start + steps >= dataset_length:
        raise ValueError(
            f"invalid segment start={start}, steps={steps}, dataset_length={dataset_length}"
        )
    return Segment(start=start, steps=steps)


def _resolve_robot_xml(args: argparse.Namespace) -> Path:
    if args.robot_xml is not None:
        return args.robot_xml
    candidates = sorted(
        (path for path in DEFAULT_FITTED_DIR.glob("*.mjcf") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No fitted MJCF found in {DEFAULT_FITTED_DIR}. Pass --robot-xml explicitly."
        )
    return candidates[0]


def _run_name(robot_xml: Path, args: argparse.Namespace, segment: Segment) -> str:
    if args.run_name:
        return args.run_name
    return (
        f"{robot_xml.stem}"
        f"_val_start{segment.start}"
        f"_steps{segment.steps}"
        f"_seed{args.seed}"
    )


def _paths(args: argparse.Namespace, run_name: str) -> tuple[Path, Path]:
    output = args.output or (args.output_dir / f"{run_name}.npz")
    csv_path = args.csv or (args.output_dir / f"{run_name}.csv")
    return output, csv_path


def _prepare_data(model, data, layout, qpos0: np.ndarray, qvel0: np.ndarray) -> None:
    mujoco = require_mujoco()
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[layout.qpos_ids] = qpos0
    data.qvel[layout.dof_ids] = qvel0
    mujoco.mj_forward(model, data)


def rollout_segment(
    model,
    layout,
    dataset,
    segment: Segment,
    *,
    sim_substeps: int,
    w_pos: float,
    w_vel: float,
    w_tau: float,
):
    mujoco = require_mujoco()
    data = mujoco.MjData(model)
    _prepare_data(
        model,
        data,
        layout,
        dataset.position[segment.start],
        dataset.velocity[segment.start],
    )

    pred_qpos = np.zeros((segment.steps, 6), dtype=np.float64)
    pred_qvel = np.zeros((segment.steps, 6), dtype=np.float64)
    pred_tau = np.zeros((segment.steps, 6), dtype=np.float64)
    target_qpos = np.zeros((segment.steps, 6), dtype=np.float64)
    target_qvel = np.zeros((segment.steps, 6), dtype=np.float64)
    target_tau = np.zeros((segment.steps, 6), dtype=np.float64)
    pos_loss = np.zeros(segment.steps, dtype=np.float64)
    vel_loss = np.zeros(segment.steps, dtype=np.float64)
    tau_loss = np.zeros(segment.steps, dtype=np.float64)
    total_loss = np.zeros(segment.steps, dtype=np.float64)

    for local_step in range(segment.steps):
        src = segment.start + local_step
        target = src + 1
        data.ctrl[:] = dataset.action_qpos[src]
        for _ in range(sim_substeps):
            mujoco.mj_step(model, data)

        pred_qpos[local_step] = data.qpos[layout.qpos_ids]
        pred_qvel[local_step] = data.qvel[layout.dof_ids]
        pred_tau[local_step] = data.qfrc_actuator[layout.dof_ids]
        target_qpos[local_step] = dataset.position[target]
        target_qvel[local_step] = dataset.velocity[target]
        target_tau[local_step] = dataset.torque[target]

        pos_loss[local_step] = float(np.mean((pred_qpos[local_step] - target_qpos[local_step]) ** 2))
        vel_loss[local_step] = float(np.mean((pred_qvel[local_step] - target_qvel[local_step]) ** 2))
        tau_loss[local_step] = float(np.mean((pred_tau[local_step] - target_tau[local_step]) ** 2))
        total_loss[local_step] = w_pos * pos_loss[local_step] + w_vel * vel_loss[local_step] + w_tau * tau_loss[local_step]

    return {
        "pred_qpos": pred_qpos,
        "pred_qvel": pred_qvel,
        "pred_tau": pred_tau,
        "target_qpos": target_qpos,
        "target_qvel": target_qvel,
        "target_tau": target_tau,
        "pos_loss": pos_loss,
        "vel_loss": vel_loss,
        "tau_loss": tau_loss,
        "total_loss": total_loss,
    }


def save_outputs(
    output: Path,
    csv_path: Path,
    *,
    args: argparse.Namespace,
    robot_xml: Path,
    segment: Segment,
    dataset,
    sim_substeps: int,
    sim_dt: float,
    results: dict[str, np.ndarray],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    frame_ids = np.arange(segment.start + 1, segment.stop + 1, dtype=np.int64)
    times = dataset.time[frame_ids]
    mean_total = float(np.mean(results["total_loss"]))
    mean_pos = float(np.mean(results["pos_loss"]))
    mean_vel = float(np.mean(results["vel_loss"]))
    mean_tau = float(np.mean(results["tau_loss"]))

    np.savez(
        output,
        robot_xml=np.asarray(str(robot_xml)),
        data_dir=np.asarray(str(args.data_dir)),
        start=np.asarray(segment.start),
        steps=np.asarray(segment.steps),
        frame_ids=frame_ids,
        times=times,
        data_dt=np.asarray(dataset.dt),
        sim_dt=np.asarray(sim_dt),
        sim_substeps=np.asarray(sim_substeps),
        w_pos=np.asarray(args.w_pos),
        w_vel=np.asarray(args.w_vel),
        w_tau=np.asarray(args.w_tau),
        mean_total_loss=np.asarray(mean_total),
        mean_pos_loss=np.asarray(mean_pos),
        mean_vel_loss=np.asarray(mean_vel),
        mean_tau_loss=np.asarray(mean_tau),
        **results,
    )

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "local_step",
                "frame_id",
                "time",
                "total_loss",
                "pos_loss",
                "vel_loss",
                "tau_loss",
            ]
        )
        for i, frame_id in enumerate(frame_ids):
            writer.writerow(
                [
                    i,
                    int(frame_id),
                    float(times[i]),
                    float(results["total_loss"][i]),
                    float(results["pos_loss"][i]),
                    float(results["vel_loss"][i]),
                    float(results["tau_loss"][i]),
                ]
            )


def play_gui(model, layout, dataset, segment: Segment, *, sim_substeps: int, speed: float, loop: bool) -> None:
    mujoco = require_mujoco()
    import mujoco.viewer

    data = mujoco.MjData(model)
    wall_dt = dataset.dt / max(speed, 1e-6)

    print("launching MuJoCo GUI; close the window to stop")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            _prepare_data(
                model,
                data,
                layout,
                dataset.position[segment.start],
                dataset.velocity[segment.start],
            )
            viewer.sync()
            time.sleep(wall_dt)

            for local_step in range(segment.steps):
                if not viewer.is_running():
                    break
                src = segment.start + local_step
                data.ctrl[:] = dataset.action_qpos[src]
                frame_start = time.time()
                for _ in range(sim_substeps):
                    mujoco.mj_step(model, data)
                viewer.sync()
                sleep_time = wall_dt - (time.time() - frame_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            if not loop:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.03)
                break


def main() -> None:
    args = parse_args()
    dataset = load_arm_dataset(args.data_dir)
    robot_xml = _resolve_robot_xml(args)
    model = load_mujoco_model(robot_xml)
    layout = arm_layout(model)

    sim_dt = float(model.opt.timestep)
    sim_substeps = max(1, int(round(dataset.dt / sim_dt)))
    effective_dt = sim_substeps * sim_dt

    steps = args.steps if args.steps is not None else _steps_from_seconds(args.duration_sec, dataset.dt)
    segment = _select_segment(dataset.length, steps, args.start, args.seed)
    run_name = _run_name(robot_xml, args, segment)
    output, csv_path = _paths(args, run_name)

    print(f"data_dir={args.data_dir}")
    print(f"robot_xml={robot_xml}")
    print(
        f"segment_start={segment.start} steps={segment.steps} "
        f"duration={segment.steps * dataset.dt:.4f}s"
    )
    print(
        f"data_dt={dataset.dt:.6g}s model_dt={sim_dt:.6g}s "
        f"substeps={sim_substeps} effective_dt={effective_dt:.6g}s"
    )

    results = rollout_segment(
        model,
        layout,
        dataset,
        segment,
        sim_substeps=sim_substeps,
        w_pos=args.w_pos,
        w_vel=args.w_vel,
        w_tau=args.w_tau,
    )
    print(
        f"mean_loss={np.mean(results['total_loss']):.6g} "
        f"pos={np.mean(results['pos_loss']):.6g} "
        f"vel={np.mean(results['vel_loss']):.6g} "
        f"tau={np.mean(results['tau_loss']):.6g}"
    )

    if args.print_every > 0:
        for i in range(0, segment.steps, args.print_every):
            print(
                f"step={i:05d} frame={segment.start + i + 1} "
                f"loss={results['total_loss'][i]:.6g} "
                f"pos={results['pos_loss'][i]:.6g} "
                f"vel={results['vel_loss'][i]:.6g} "
                f"tau={results['tau_loss'][i]:.6g}"
            )

    if not args.no_save:
        save_outputs(
            output,
            csv_path,
            args=args,
            robot_xml=robot_xml,
            segment=segment,
            dataset=dataset,
            sim_substeps=sim_substeps,
            sim_dt=sim_dt,
            results=results,
        )
        print(f"saved validation npz: {output}")
        print(f"saved per-step csv: {csv_path}")

    if args.gui:
        play_gui(
            model,
            layout,
            dataset,
            segment,
            sim_substeps=sim_substeps,
            speed=args.speed,
            loop=args.loop,
        )


if __name__ == "__main__":
    main()
