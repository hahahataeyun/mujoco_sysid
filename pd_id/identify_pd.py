from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from pd_id.data import (
    apply_warmup_mask,
    full_sequence_batch,
    load_arm_dataset,
    sample_fixed_window_batch,
)
from pd_id.model import initial_parameters, load_mujoco_model, arm_layout, write_fitted_xml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/home/capture15/shared_data/capture/test_pd/ik/6/raw/arm")
DEFAULT_INITIAL_MJCF_DIR = REPO_ROOT / "pd_id/initial_mjcf"
DEFAULT_INITIAL_MODEL = "xarm_model_0"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pd_id_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify xArm actuator and joint parameters from recorded PD data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--robot-xml", type=Path, default=None)
    parser.add_argument("--initial-mjcf-dir", type=Path, default=DEFAULT_INITIAL_MJCF_DIR)
    parser.add_argument("--initial-model", type=str, default=DEFAULT_INITIAL_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fitted-xml", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-fitted-xml", action="store_true")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of initial rollout steps to ignore in the training loss.",
    )
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--grad-mode", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w-pos", type=float, default=100.0)
    parser.add_argument("--w-vel", type=float, default=10.0)
    parser.add_argument("--w-tau", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--positive-floor", type=float, default=1e-8)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--no-final-rollout", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="pd-id")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def _steps_from_seconds(seconds: float, dt: float) -> int:
    return max(1, int(round(seconds / dt)))


def _host_metrics(metrics) -> dict[str, float | np.ndarray]:
    host = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        host[key] = float(array) if array.ndim == 0 else array.astype(np.float64)
    return host


def _wandb_metrics(metrics: dict[str, float | np.ndarray]) -> dict[str, float]:
    log_data: dict[str, float] = {
        "loss/total": float(metrics["loss"]),
        "loss/position": float(metrics["pos_loss"]),
        "loss/velocity": float(metrics["vel_loss"]),
        "loss/torque": float(metrics["tau_loss"]),
        "optim/grad_norm": float(metrics["grad_norm"]),
    }

    joint_metric_names = {
        "joint/total_loss": "loss_joint/total",
        "joint/pos_loss": "loss_joint/position",
        "joint/vel_loss": "loss_joint/velocity",
        "joint/tau_loss": "loss_joint/torque",
    }
    for source_key, wandb_prefix in joint_metric_names.items():
        values = np.asarray(metrics[source_key], dtype=np.float64)
        for joint_id, value in enumerate(values, start=1):
            log_data[f"{wandb_prefix}/joint{joint_id}"] = float(value)
    return log_data


def _format_seconds(seconds: float) -> str:
    text = f"{seconds:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _resolve_robot_xml(args: argparse.Namespace) -> Path:
    if args.robot_xml is not None:
        return args.robot_xml
    initial_model = args.initial_model
    if not initial_model.endswith(".mjcf"):
        initial_model = f"{initial_model}.mjcf"
    return args.initial_mjcf_dir / initial_model


def _default_run_name(
    args: argparse.Namespace, robot_xml: Path, window_steps: int
) -> str:
    robot_stem = robot_xml.stem
    window = _format_seconds(args.window_sec)
    lr = f"{args.lr:.0e}".replace("+", "").replace("-", "m")
    return (
        f"{robot_stem}_pd"
        f"_bs{args.batch_size}"
        f"_win{window}s-{window_steps}step"
        f"_warm{args.warmup_steps}"
        f"_iter{args.iters}"
        f"_lr{lr}"
        f"_seed{args.seed}"
    )


def _resolve_output_paths(
    args: argparse.Namespace, robot_xml: Path, window_steps: int
) -> tuple[Path, Path, Path | None, str]:
    run_name = args.run_name or _default_run_name(args, robot_xml, window_steps)
    run_dir = args.output_dir / run_name
    output = args.output or (run_dir / "result.npz")
    fitted_xml = None if args.no_fitted_xml else (args.fitted_xml or run_dir / "fitted_xarm_final.mjcf")
    return run_dir, output, fitted_xml, run_name


def _jsonify(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _write_run_config(
    path: Path,
    *,
    args: argparse.Namespace,
    run_name: str,
    run_dir: Path,
    robot_xml: Path,
    output_path: Path,
    fitted_xml_path: Path | None,
    data_dt: float,
    model_dt_original: float,
    model_dt_effective: float,
    effective_dt: float,
    sim_substeps: int,
    window_steps: int,
) -> None:
    config = {
        "run_name": run_name,
        "run_dir": run_dir,
        "initial_model": robot_xml.stem,
        "initial_mjcf": robot_xml,
        "data_dir": args.data_dir,
        "output": output_path,
        "fitted_xml": fitted_xml_path,
        "args": vars(args),
        "derived": {
            "data_dt": data_dt,
            "model_dt_original": model_dt_original,
            "model_dt_effective": model_dt_effective,
            "effective_dt": effective_dt,
            "sim_substeps": sim_substeps,
            "window_steps": window_steps,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonify(config), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _init_wandb(
    args: argparse.Namespace,
    *,
    run_name: str,
    robot_xml: Path,
    output_path: Path,
    fitted_xml_path: Path | None,
    data_dt: float,
    model_dt_original: float,
    model_dt_effective: float,
    sim_substeps: int,
    window_steps: int,
    warmup_steps: int,
):
    if not args.wandb:
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb logging was requested with --wandb, but wandb is not installed. "
            "Install it with `pip install wandb` or run without --wandb."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        mode=args.wandb_mode,
        config={
            "data_dir": str(args.data_dir),
            "initial_mjcf": str(robot_xml),
            "output": str(output_path),
            "fitted_xml": None if fitted_xml_path is None else str(fitted_xml_path),
            "checkpoint_every": args.checkpoint_every,
            "iters": args.iters,
            "batch_size": args.batch_size,
            "window_sec": args.window_sec,
            "window_steps": window_steps,
            "warmup_steps": warmup_steps,
            "lr": args.lr,
            "grad_mode": args.grad_mode,
            "seed": args.seed,
            "w_pos": args.w_pos,
            "w_vel": args.w_vel,
            "w_tau": args.w_tau,
            "grad_clip_norm": args.grad_clip_norm,
            "positive_floor": args.positive_floor,
            "data_dt": data_dt,
            "model_dt_original": model_dt_original,
            "model_dt_effective": model_dt_effective,
            "sim_substeps": sim_substeps,
        },
    )


def _save_wandb_outputs(
    wandb_run,
    *,
    run_name: str,
    run_dir: Path,
    output_path: Path,
    fitted_xml_path: Path | None,
    config_path: Path,
) -> None:
    if wandb_run is None:
        return
    try:
        import wandb
    except ModuleNotFoundError:
        return

    artifact = wandb.Artifact(
        name=f"{run_name}-outputs",
        type="pd_id_output",
        metadata={
            "result_npz": str(output_path),
            "fitted_xml": None if fitted_xml_path is None else str(fitted_xml_path),
            "config": str(config_path),
        },
    )
    if config_path.exists():
        artifact.add_file(str(config_path), name=config_path.name)
    if output_path.exists():
        artifact.add_file(str(output_path), name=output_path.name)
    if fitted_xml_path is not None and fitted_xml_path.exists():
        artifact.add_file(str(fitted_xml_path), name=fitted_xml_path.name)
    for checkpoint_path in sorted(run_dir.glob("fitted_xarm_*.mjcf")):
        if fitted_xml_path is not None and checkpoint_path.resolve() == fitted_xml_path.resolve():
            continue
        artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    wandb_run.log_artifact(artifact)
    wandb_run.config.update(
        {
            "output_artifact": artifact.name,
            "output_npz": str(output_path),
            "output_fitted_xml": None if fitted_xml_path is None else str(fitted_xml_path),
            "output_config": str(config_path),
        },
        allow_val_change=True,
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("JAX_ENABLE_X64", "false")

    from pd_id.mjx_optimizer import (
        LossWeights,
        TrainConfig,
        batch_to_jax,
        build_trainer,
        init_raw_params,
        materialize_params_np,
    )

    robot_xml = _resolve_robot_xml(args)

    dataset = load_arm_dataset(args.data_dir)
    mujoco_model = load_mujoco_model(robot_xml)

    original_sim_dt = float(mujoco_model.opt.timestep)
    sim_substeps = max(1, int(round(dataset.dt / original_sim_dt)))
    sim_dt = dataset.dt / sim_substeps
    mujoco_model.opt.timestep = sim_dt
    effective_dt = sim_substeps * sim_dt
    window_steps = _steps_from_seconds(args.window_sec, dataset.dt)
    if args.warmup_steps < 0:
        raise ValueError(f"--warmup-steps must be non-negative, got {args.warmup_steps}")
    if args.warmup_steps >= window_steps:
        raise ValueError(
            f"--warmup-steps ({args.warmup_steps}) must be smaller than "
            f"window_steps ({window_steps}). Increase --window-sec or reduce warmup."
        )
    run_dir, output_path, fitted_xml_path, run_name = _resolve_output_paths(
        args, robot_xml, window_steps
    )
    config_path = run_dir / "config.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_every < 0:
        raise ValueError(f"--checkpoint-every must be non-negative, got {args.checkpoint_every}")

    layout = arm_layout(mujoco_model)
    initial = initial_parameters(mujoco_model, layout)

    print(f"data_dir={args.data_dir}")
    print(f"initial_mjcf={robot_xml}")
    print(f"run_name={run_name}")
    print(
        f"samples={dataset.length} data_dt={dataset.dt:.6g}s "
        f"model_dt_original={original_sim_dt:.6g}s "
        f"model_dt={sim_dt:.6g}s substeps={sim_substeps} effective_dt={effective_dt:.6g}s"
    )
    print(
        f"batch_size={args.batch_size} window_sec={args.window_sec:g} "
        f"window_steps={window_steps} warmup_steps={args.warmup_steps} "
        f"loss_weights=(pos={args.w_pos}, vel={args.w_vel}, tau={args.w_tau})"
    )
    print(f"run_dir={run_dir}")
    print(f"output={output_path}")
    print(f"config={config_path}")
    if fitted_xml_path is not None:
        print(f"fitted_xml={fitted_xml_path}")
    print(f"grad_mode={args.grad_mode}")

    _write_run_config(
        config_path,
        args=args,
        run_name=run_name,
        run_dir=run_dir,
        robot_xml=robot_xml,
        output_path=output_path,
        fitted_xml_path=fitted_xml_path,
        data_dt=dataset.dt,
        model_dt_original=original_sim_dt,
        model_dt_effective=sim_dt,
        effective_dt=effective_dt,
        sim_substeps=sim_substeps,
        window_steps=window_steps,
    )

    wandb_run = _init_wandb(
        args,
        run_name=run_name,
        robot_xml=robot_xml,
        output_path=output_path,
        fitted_xml_path=fitted_xml_path,
        data_dt=dataset.dt,
        model_dt_original=original_sim_dt,
        model_dt_effective=sim_dt,
        sim_substeps=sim_substeps,
        window_steps=window_steps,
        warmup_steps=args.warmup_steps,
    )

    raw_params = init_raw_params(initial, args.positive_floor)
    weights = LossWeights(position=args.w_pos, velocity=args.w_vel, torque=args.w_tau)
    config = TrainConfig(
        sim_substeps=sim_substeps,
        learning_rate=args.lr,
        grad_mode=args.grad_mode,
        grad_clip_norm=args.grad_clip_norm,
        positive_floor=args.positive_floor,
    )
    adam_init, train_step, evaluate, rollout = build_trainer(
        mujoco_model, layout, initial, weights, config
    )
    state = adam_init(raw_params)

    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []

    for iteration in range(1, args.iters + 1):
        batch = sample_fixed_window_batch(
            dataset,
            batch_size=args.batch_size,
            steps=window_steps,
            rng=rng,
        )
        batch = apply_warmup_mask(batch, args.warmup_steps)
        state, metrics = train_step(state, batch_to_jax(batch))
        metrics_host = _host_metrics(metrics)
        history.append(metrics_host)
        if wandb_run is not None:
            log_data = _wandb_metrics(metrics_host)
            log_data["iteration"] = iteration
            wandb_run.log(log_data, step=iteration)

        if (
            args.print_every > 0
            and (iteration == 1 or iteration % args.print_every == 0 or iteration == args.iters)
        ):
            print(
                f"iter={iteration:05d} loss={metrics_host['loss']:.6g} "
                f"pos={metrics_host['pos_loss']:.6g} vel={metrics_host['vel_loss']:.6g} "
                f"tau={metrics_host['tau_loss']:.6g} grad={metrics_host['grad_norm']:.6g}"
            )
        if (
            fitted_xml_path is not None
            and args.checkpoint_every > 0
            and iteration % args.checkpoint_every == 0
        ):
            checkpoint_params = materialize_params_np(state["params"], args.positive_floor)
            checkpoint_path = run_dir / f"fitted_xarm_{iteration}.mjcf"
            write_fitted_xml(
                robot_xml, checkpoint_path, params=checkpoint_params, timestep=sim_dt
            )
            print(f"saved checkpoint xml: {checkpoint_path}")

    final_params = materialize_params_np(state["params"], args.positive_floor)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_payload = {
        "history": np.asarray(
            [
                [
                    row["loss"],
                    row["pos_loss"],
                    row["vel_loss"],
                    row["tau_loss"],
                    row["grad_norm"],
                ]
                for row in history
            ],
            dtype=np.float64,
        ),
        "history_columns": np.asarray(
            ["loss", "pos_loss", "vel_loss", "tau_loss", "grad_norm"]
        ),
        "data_dir": np.asarray(str(args.data_dir)),
        "robot_xml": np.asarray(str(robot_xml)),
        "initial_mjcf": np.asarray(str(robot_xml)),
        "initial_model": np.asarray(robot_xml.stem),
        "run_dir": np.asarray(str(run_dir)),
        "config_path": np.asarray(str(config_path)),
        "sim_substeps": np.asarray(sim_substeps),
        "data_dt": np.asarray(dataset.dt),
        "model_dt_original": np.asarray(original_sim_dt),
        "model_dt": np.asarray(sim_dt),
        "window_sec": np.asarray(args.window_sec),
        "window_steps": np.asarray(window_steps),
        "warmup_steps": np.asarray(args.warmup_steps),
        "batch_size": np.asarray(args.batch_size),
        "checkpoint_every": np.asarray(args.checkpoint_every),
        "iters": np.asarray(args.iters),
        "seed": np.asarray(args.seed),
        "run_name": np.asarray(run_name),
        **final_params,
    }

    if not args.no_final_rollout:
        full_batch = full_sequence_batch(dataset)
        pred_qpos, pred_qvel, pred_tau = rollout(state["params"], batch_to_jax(full_batch))
        sim_position = np.vstack([dataset.position[:1], np.asarray(pred_qpos[0])])
        sim_velocity = np.vstack([dataset.velocity[:1], np.asarray(pred_qvel[0])])
        sim_torque = np.vstack([np.zeros((1, 6), dtype=np.float64), np.asarray(pred_tau[0])])
        full_metrics = _host_metrics(evaluate(state["params"], batch_to_jax(full_batch)))
        print(
            f"full_sequence loss={full_metrics['loss']:.6g} "
            f"pos={full_metrics['pos_loss']:.6g} vel={full_metrics['vel_loss']:.6g} "
            f"tau={full_metrics['tau_loss']:.6g}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "full_sequence/loss_total": full_metrics["loss"],
                    "full_sequence/loss_position": full_metrics["pos_loss"],
                    "full_sequence/loss_velocity": full_metrics["vel_loss"],
                    "full_sequence/loss_torque": full_metrics["tau_loss"],
                },
                step=args.iters,
            )
        save_payload.update(
            {
                "sim_position": sim_position,
                "sim_velocity": sim_velocity,
                "sim_torque": sim_torque,
                "full_loss": np.asarray(full_metrics["loss"]),
                "full_pos_loss": np.asarray(full_metrics["pos_loss"]),
                "full_vel_loss": np.asarray(full_metrics["vel_loss"]),
                "full_tau_loss": np.asarray(full_metrics["tau_loss"]),
            }
        )

    np.savez(output_path, **save_payload)
    print(f"saved result: {output_path}")

    if fitted_xml_path is not None:
        write_fitted_xml(
            robot_xml, fitted_xml_path, params=final_params, timestep=sim_dt
        )
        print(f"saved fitted xml: {fitted_xml_path}")

    _save_wandb_outputs(
        wandb_run,
        run_name=run_name,
        run_dir=run_dir,
        output_path=output_path,
        fitted_xml_path=fitted_xml_path,
        config_path=config_path,
    )

    for key, value in final_params.items():
        print(f"{key}: {np.array2string(value, precision=6, floatmode='fixed')}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
