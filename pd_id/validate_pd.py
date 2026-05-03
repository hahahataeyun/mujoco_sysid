'''
    fitted    = fitted MJCF + action_qpos dynamics rollout
    unfitted  = initial MJCF + action_qpos dynamics rollout
    kinematic = measured position.npy replay
'''

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

try:
    from pd_id.data import load_arm_dataset
    from pd_id.model import (
        ARM_JOINTS,
        DEFAULT_MESH_ROOTS,
        _resolve_asset_file,
        arm_layout,
        load_mujoco_model,
        require_mujoco,
    )
except ModuleNotFoundError:
    from data import load_arm_dataset
    from model import (
        ARM_JOINTS,
        DEFAULT_MESH_ROOTS,
        _resolve_asset_file,
        arm_layout,
        load_mujoco_model,
        require_mujoco,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/home/capture15/shared_data/capture/test_pd/ik/6/raw/arm")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pd_id_results/validation"
DEFAULT_FITTED_DIR = REPO_ROOT / "pd_id_results"
DEFAULT_INITIAL_MJCF_DIR = REPO_ROOT / "pd_id/initial_mjcf"
UNFITTED_JOINTS = tuple(f"unfit_joint{i}" for i in range(1, 7))
FITTED_KINEMATIC_JOINTS = tuple(f"fit_kin_joint{i}" for i in range(1, 7))
UNFITTED_KINEMATIC_JOINTS = tuple(f"unfit_kin_joint{i}" for i in range(1, 7))
DYNAMIC_OVERLAY_RGBA = "1 1 1 0.35"


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
    parser.add_argument("--unfitted-xml", type=Path, default=None)
    parser.add_argument("--initial-mjcf-dir", type=Path, default=DEFAULT_INITIAL_MJCF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w-pos", type=float, default=100.0)
    parser.add_argument("--w-vel", type=float, default=10.0)
    parser.add_argument("--w-tau", type=float, default=1.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--compare-kinematic", action="store_true")
    parser.add_argument("--fitted-offset", type=float, nargs=3, default=(-0.6, 0.0, 0.0))
    parser.add_argument("--unfitted-offset", type=float, nargs=3, default=(0.6, 0.0, 0.0))
    parser.add_argument(
        "--kinematic-offset",
        type=float,
        nargs=3,
        default=None,
        help="Deprecated; kinematic replays are overlaid at fitted/unfitted offsets.",
    )
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


def _resolve_unfitted_xml(args: argparse.Namespace, fitted_xml: Path) -> Path:
    if args.unfitted_xml is not None:
        return args.unfitted_xml
    initial_stem = fitted_xml.stem.split("_pd_")[0]
    candidate = args.initial_mjcf_dir / f"{initial_stem}.mjcf"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not infer unfitted MJCF from fitted model {fitted_xml}. "
        f"Tried {candidate}. Pass --unfitted-xml explicitly."
    )


def _run_name(robot_xml: Path, args: argparse.Namespace, segment: Segment) -> str:
    if args.run_name:
        return args.run_name
    return (
        f"{robot_xml.stem}"
        f"_val_start{segment.start}"
        f"_steps{segment.steps}"
        f"_seed{args.seed}"
    )


def _paths(args: argparse.Namespace, run_name: str) -> tuple[Path, Path, Path | None]:
    output = args.output or (args.output_dir / f"{run_name}.npz")
    plot_path = args.plot or (args.output_dir / f"{run_name}.png")
    return output, plot_path, args.csv


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


def _prefixed_results(prefix: str, results: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}_{key}": value for key, value in results.items()}


def plot_losses(
    plot_path: Path,
    *,
    times: np.ndarray,
    fitted: dict[str, np.ndarray],
    unfitted: dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to save validation plots. Install it with "
            "`pip install matplotlib`."
        ) from exc

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    t = times - times[0]
    panels = [
        ("total_loss", "total"),
        ("pos_loss", "position"),
        ("vel_loss", "velocity"),
        ("tau_loss", "torque"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 9), sharex=True)
    for axis, (key, title) in zip(axes, panels):
        axis.plot(t, fitted[key], label="fitted", linewidth=1.5)
        axis.plot(t, unfitted[key], label="unfitted", linewidth=1.2, alpha=0.85)
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("time from segment start (s)")
    fig.suptitle("Validation loss per timestep")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_outputs(
    output: Path,
    plot_path: Path,
    *,
    args: argparse.Namespace,
    fitted_xml: Path,
    unfitted_xml: Path,
    segment: Segment,
    dataset,
    sim_substeps: int,
    model_dt_original: float,
    sim_dt: float,
    fitted_results: dict[str, np.ndarray],
    unfitted_results: dict[str, np.ndarray],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_ids = np.arange(segment.start + 1, segment.stop + 1, dtype=np.int64)
    times = dataset.time[frame_ids]

    plot_losses(plot_path, times=times, fitted=fitted_results, unfitted=unfitted_results)

    np.savez(
        output,
        fitted_xml=np.asarray(str(fitted_xml)),
        unfitted_xml=np.asarray(str(unfitted_xml)),
        data_dir=np.asarray(str(args.data_dir)),
        start=np.asarray(segment.start),
        steps=np.asarray(segment.steps),
        frame_ids=frame_ids,
        times=times,
        data_dt=np.asarray(dataset.dt),
        model_dt_original=np.asarray(model_dt_original),
        sim_dt=np.asarray(sim_dt),
        sim_substeps=np.asarray(sim_substeps),
        w_pos=np.asarray(args.w_pos),
        w_vel=np.asarray(args.w_vel),
        w_tau=np.asarray(args.w_tau),
        fitted_mean_total_loss=np.asarray(float(np.mean(fitted_results["total_loss"]))),
        fitted_mean_pos_loss=np.asarray(float(np.mean(fitted_results["pos_loss"]))),
        fitted_mean_vel_loss=np.asarray(float(np.mean(fitted_results["vel_loss"]))),
        fitted_mean_tau_loss=np.asarray(float(np.mean(fitted_results["tau_loss"]))),
        unfitted_mean_total_loss=np.asarray(float(np.mean(unfitted_results["total_loss"]))),
        unfitted_mean_pos_loss=np.asarray(float(np.mean(unfitted_results["pos_loss"]))),
        unfitted_mean_vel_loss=np.asarray(float(np.mean(unfitted_results["vel_loss"]))),
        unfitted_mean_tau_loss=np.asarray(float(np.mean(unfitted_results["tau_loss"]))),
        **_prefixed_results("fitted", fitted_results),
        **_prefixed_results("unfitted", unfitted_results),
    )

    if args.csv is None:
        return

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "local_step",
                "frame_id",
                "time",
                "fitted_total_loss",
                "unfitted_total_loss",
                "fitted_pos_loss",
                "unfitted_pos_loss",
                "fitted_vel_loss",
                "unfitted_vel_loss",
                "fitted_tau_loss",
                "unfitted_tau_loss",
            ]
        )
        for i, frame_id in enumerate(frame_ids):
            writer.writerow(
                [
                    i,
                    int(frame_id),
                    float(times[i]),
                    float(fitted_results["total_loss"][i]),
                    float(unfitted_results["total_loss"][i]),
                    float(fitted_results["pos_loss"][i]),
                    float(unfitted_results["pos_loss"][i]),
                    float(fitted_results["vel_loss"][i]),
                    float(unfitted_results["vel_loss"][i]),
                    float(fitted_results["tau_loss"][i]),
                    float(unfitted_results["tau_loss"][i]),
                ]
            )


def _parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(value) for value in text.split()], dtype=np.float64)


def _format_vec(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _style_geoms(
    elem: ET.Element,
    *,
    rgba: str | None = None,
    disable_collision: bool = False,
) -> None:
    for node in elem.iter("geom"):
        if rgba is not None:
            node.attrib.pop("material", None)
            node.set("rgba", rgba)
        if disable_collision:
            node.set("contype", "0")
            node.set("conaffinity", "0")


def _prefix_tree_names(elem: ET.Element, prefix: str) -> None:
    for node in elem.iter():
        name = node.get("name")
        if name:
            node.set("name", f"{prefix}{name}")


def _prefix_actuator(elem: ET.Element, prefix: str) -> None:
    for node in elem.iter():
        name = node.get("name")
        if name:
            node.set("name", f"{prefix}{name}")
        joint = node.get("joint")
        if joint:
            node.set("joint", f"{prefix}{joint}")


def _resolve_xml_assets(root: ET.Element, xml_dir: Path) -> None:
    for elem in root.findall(".//*[@file]"):
        file_name = elem.get("file")
        if file_name:
            elem.set("file", str(_resolve_asset_file(file_name, xml_dir, DEFAULT_MESH_ROOTS)))


def _append_label(worldbody: ET.Element, name: str, pos: np.ndarray) -> None:
    label_body = ET.SubElement(worldbody, "body", {"name": f"label_{name}", "pos": _format_vec(pos)})
    ET.SubElement(label_body, "site", {"name": f"<{name}>", "type": "sphere", "size": "0.035", "rgba": "1 1 0 1"})


def _append_prefixed_model(
    worldbody: ET.Element,
    actuator_root: ET.Element,
    source_root: ET.Element,
    *,
    prefix: str,
    offset: np.ndarray,
    include_actuators: bool,
    geom_rgba: str | None = None,
    disable_collision: bool = True,
) -> None:
    source_worldbody = source_root.find("worldbody")
    if source_worldbody is None:
        raise ValueError("source model has no <worldbody>")
    for child in list(source_worldbody):
        if child.tag not in {"body", "geom"}:
            continue
        copied = copy.deepcopy(child)
        _prefix_tree_names(copied, prefix)
        _style_geoms(copied, rgba=geom_rgba, disable_collision=disable_collision)
        pos = _parse_vec(copied.get("pos"), (0.0, 0.0, 0.0))
        copied.set("pos", _format_vec(pos + offset))
        worldbody.append(copied)

    if not include_actuators:
        return
    source_actuator = source_root.find("actuator")
    if source_actuator is None:
        return
    for child in list(source_actuator):
        copied = copy.deepcopy(child)
        _prefix_actuator(copied, prefix)
        actuator_root.append(copied)


def _create_compare_xml(
    fitted_xml: Path,
    unfitted_xml: Path,
    *,
    fitted_offset: np.ndarray,
    unfitted_offset: np.ndarray,
) -> Path:
    tree = ET.parse(fitted_xml)
    root = tree.getroot()
    root.set("model", f"{root.get('model', 'xarm')}_compare")
    _resolve_xml_assets(root, fitted_xml.parent)
    fitted_source_root = copy.deepcopy(root)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{fitted_xml} has no <worldbody>")
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")

    for child in list(worldbody):
        if child.tag not in {"body", "geom"}:
            continue
        pos = _parse_vec(child.get("pos"), (0.0, 0.0, 0.0))
        child.set("pos", _format_vec(pos + fitted_offset))
        _style_geoms(child, rgba=DYNAMIC_OVERLAY_RGBA, disable_collision=False)

    unfitted_tree = ET.parse(unfitted_xml)
    unfitted_root = unfitted_tree.getroot()
    _resolve_xml_assets(unfitted_root, unfitted_xml.parent)

    _append_prefixed_model(
        worldbody,
        actuator,
        unfitted_root,
        prefix="unfit_",
        offset=unfitted_offset,
        include_actuators=True,
        geom_rgba=DYNAMIC_OVERLAY_RGBA,
        disable_collision=False,
    )

    _append_prefixed_model(
        worldbody,
        actuator,
        fitted_source_root,
        prefix="fit_kin_",
        offset=fitted_offset,
        include_actuators=False,
        geom_rgba=None,
        disable_collision=True,
    )

    _append_prefixed_model(
        worldbody,
        actuator,
        unfitted_root,
        prefix="unfit_kin_",
        offset=unfitted_offset,
        include_actuators=False,
        geom_rgba=None,
        disable_collision=True,
    )

    _append_label(worldbody, "fitted", fitted_offset + np.array([0.0, 0.0, 1.1]))
    _append_label(worldbody, "unfitted", unfitted_offset + np.array([0.0, 0.0, 1.1]))

    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".mjcf", prefix=f"{fitted_xml.stem}_compare_", delete=False
    )
    with handle:
        tree.write(handle, encoding="unicode", xml_declaration=False)
    return Path(handle.name)


def _set_layout_state(model, data, layout, qpos: np.ndarray, qvel: np.ndarray) -> None:
    mujoco = require_mujoco()
    data.qpos[layout.qpos_ids] = qpos
    data.qvel[layout.dof_ids] = qvel
    mujoco.mj_forward(model, data)


def play_gui(
    model,
    fitted_layout,
    dataset,
    segment: Segment,
    *,
    sim_substeps: int,
    speed: float,
    loop: bool,
    unfitted_layout=None,
    kinematic_layouts=(),
) -> None:
    mujoco = require_mujoco()
    import mujoco.viewer

    data = mujoco.MjData(model)
    wall_dt = dataset.dt / max(speed, 1e-6)
    reset_requested = {"value": False}

    def key_callback(keycode: int) -> None:
        if keycode in (ord("R"), ord("r")):
            reset_requested["value"] = True

    def reset_scene() -> None:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        _set_layout_state(
            model,
            data,
            fitted_layout,
            dataset.position[segment.start],
            dataset.velocity[segment.start],
        )
        if unfitted_layout is not None:
            _set_layout_state(
                model,
                data,
                unfitted_layout,
                dataset.position[segment.start],
                dataset.velocity[segment.start],
            )
        for kinematic_layout in kinematic_layouts:
            _set_layout_state(
                model,
                data,
                kinematic_layout,
                dataset.position[segment.start],
                dataset.velocity[segment.start],
            )

    if unfitted_layout is not None or kinematic_layouts:
        print(
            "launching MuJoCo GUI; fitted/unfitted dynamics are transparent over kinematic replays; press R to restart"
        )
    else:
        print("launching MuJoCo GUI; press R to restart, close the window to stop")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        try:
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        except Exception:
            pass
        while viewer.is_running():
            reset_requested["value"] = False
            reset_scene()
            viewer.sync()
            time.sleep(wall_dt)

            for local_step in range(segment.steps):
                if not viewer.is_running() or reset_requested["value"]:
                    break
                src = segment.start + local_step
                data.ctrl[:] = 0.0
                data.ctrl[fitted_layout.actuator_ids] = dataset.action_qpos[src]
                if unfitted_layout is not None:
                    data.ctrl[unfitted_layout.actuator_ids] = dataset.action_qpos[src]
                frame_start = time.time()
                for _ in range(sim_substeps):
                    mujoco.mj_step(model, data)
                if kinematic_layouts:
                    target = min(src + 1, dataset.length - 1)
                    for kinematic_layout in kinematic_layouts:
                        _set_layout_state(
                            model,
                            data,
                            kinematic_layout,
                            dataset.position[target],
                            dataset.velocity[target],
                        )
                viewer.sync()
                sleep_time = wall_dt - (time.time() - frame_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            if reset_requested["value"]:
                continue
            if not loop:
                while viewer.is_running():
                    if reset_requested["value"]:
                        break
                    viewer.sync()
                    time.sleep(0.03)
                if reset_requested["value"]:
                    continue
                break


def main() -> None:
    args = parse_args()
    dataset = load_arm_dataset(args.data_dir)
    robot_xml = _resolve_robot_xml(args)
    unfitted_xml = _resolve_unfitted_xml(args, robot_xml)
    model = load_mujoco_model(robot_xml)
    layout = arm_layout(model)
    unfitted_model = load_mujoco_model(unfitted_xml)

    model_dt_original = float(model.opt.timestep)
    sim_substeps = max(1, int(round(dataset.dt / model_dt_original)))
    sim_dt = dataset.dt / sim_substeps
    model.opt.timestep = sim_dt
    effective_dt = sim_substeps * sim_dt
    unfitted_model.opt.timestep = sim_dt

    steps = args.steps if args.steps is not None else _steps_from_seconds(args.duration_sec, dataset.dt)
    segment = _select_segment(dataset.length, steps, args.start, args.seed)
    run_name = _run_name(robot_xml, args, segment)
    output, plot_path, csv_path = _paths(args, run_name)

    print(f"data_dir={args.data_dir}")
    print(f"fitted_xml={robot_xml}")
    print(f"unfitted_xml={unfitted_xml}")
    print(
        f"segment_start={segment.start} steps={segment.steps} "
        f"duration={segment.steps * dataset.dt:.4f}s"
    )
    print(
        f"data_dt={dataset.dt:.6g}s model_dt_original={model_dt_original:.6g}s "
        f"model_dt={sim_dt:.6g}s "
        f"substeps={sim_substeps} effective_dt={effective_dt:.6g}s"
    )

    fitted_results = rollout_segment(
        model,
        layout,
        dataset,
        segment,
        sim_substeps=sim_substeps,
        w_pos=args.w_pos,
        w_vel=args.w_vel,
        w_tau=args.w_tau,
    )
    unfitted_results = rollout_segment(
        unfitted_model,
        arm_layout(unfitted_model),
        dataset,
        segment,
        sim_substeps=sim_substeps,
        w_pos=args.w_pos,
        w_vel=args.w_vel,
        w_tau=args.w_tau,
    )
    print(
        f"fitted_mean_loss={np.mean(fitted_results['total_loss']):.6g} "
        f"unfitted_mean_loss={np.mean(unfitted_results['total_loss']):.6g}"
    )

    if args.print_every > 0:
        for i in range(0, segment.steps, args.print_every):
            print(
                f"step={i:05d} frame={segment.start + i + 1} "
                f"fitted={fitted_results['total_loss'][i]:.6g} "
                f"unfitted={unfitted_results['total_loss'][i]:.6g}"
            )

    if not args.no_save:
        save_outputs(
            output,
            plot_path,
            args=args,
            fitted_xml=robot_xml,
            unfitted_xml=unfitted_xml,
            segment=segment,
            dataset=dataset,
            sim_substeps=sim_substeps,
            model_dt_original=model_dt_original,
            sim_dt=sim_dt,
            fitted_results=fitted_results,
            unfitted_results=unfitted_results,
        )
        print(f"saved validation npz: {output}")
        print(f"saved validation plot: {plot_path}")
        if csv_path is not None:
            print(f"saved per-step csv: {csv_path}")

    if args.gui:
        gui_model = model
        gui_layout = layout
        unfitted_layout = None
        kinematic_layouts = ()
        if args.compare_kinematic:
            compare_xml = _create_compare_xml(
                robot_xml,
                unfitted_xml,
                fitted_offset=np.asarray(args.fitted_offset, dtype=np.float64),
                unfitted_offset=np.asarray(args.unfitted_offset, dtype=np.float64),
            )
            print(f"compare_gui_xml={compare_xml}")
            gui_model = load_mujoco_model(compare_xml)
            gui_layout = arm_layout(gui_model, ARM_JOINTS)
            unfitted_layout = arm_layout(gui_model, UNFITTED_JOINTS)
            kinematic_layouts = (
                arm_layout(
                    gui_model, FITTED_KINEMATIC_JOINTS, require_actuators=False
                ),
                arm_layout(
                    gui_model, UNFITTED_KINEMATIC_JOINTS, require_actuators=False
                ),
            )

        play_gui(
            gui_model,
            gui_layout,
            dataset,
            segment,
            sim_substeps=sim_substeps,
            speed=args.speed,
            loop=args.loop,
            unfitted_layout=unfitted_layout,
            kinematic_layouts=kinematic_layouts,
        )


if __name__ == "__main__":
    main()
