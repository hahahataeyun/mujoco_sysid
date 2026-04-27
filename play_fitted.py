"""Replay a fitted book friction result in the MuJoCo GUI."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from prototype import (
    DEFAULT_OBJECT_URDF,
    DEFAULT_ROBOT_XML,
    DEFAULT_SAVE_PATH,
    DEFAULT_SEQUENCE_DIR,
    build_scene,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--object-urdf", type=Path, default=DEFAULT_OBJECT_URDF)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--contact-mode", choices=("tips", "hand", "all"), default="tips")
    parser.add_argument("--hand-command-min", type=float, default=1000.0)
    parser.add_argument("--hand-command-max", type=float, default=2000.0)
    parser.add_argument("--apply-c2r", dest="apply_c2r", action="store_true", default=None)
    parser.add_argument("--no-apply-c2r", dest="apply_c2r", action="store_false")
    parser.add_argument("--no-support-plane", action="store_true")
    parser.add_argument("--support-plane-z", type=float, default=None)
    return parser.parse_args()


def infer_frame_slice(frame_ids: np.ndarray) -> tuple[int, int, int]:
    if len(frame_ids) < 2:
        return int(frame_ids[0]), 1, len(frame_ids)
    diffs = np.diff(frame_ids)
    stride = int(round(float(np.median(diffs))))
    stride = max(stride, 1)
    return int(frame_ids[0]), stride, len(frame_ids)


def make_scene_args(
    args: argparse.Namespace, friction: np.ndarray, frame_ids: np.ndarray, apply_c2r: bool
):
    result_start, result_stride, result_count = infer_frame_slice(frame_ids)

    return argparse.Namespace(
        sequence_dir=args.sequence_dir,
        robot_xml=args.robot_xml,
        object_urdf=args.object_urdf,
        save_path=args.result,
        start_frame=args.start_frame if args.start_frame is not None else result_start,
        max_frames=args.max_frames if args.max_frames is not None else result_count,
        stride=args.stride if args.stride is not None else result_stride,
        iters=0,
        lr=0.0,
        backend="mujoco",
        fd_eps=1e-2,
        hand_command_min=args.hand_command_min,
        hand_command_max=args.hand_command_max,
        rot_weight=0.02,
        contact_mode=args.contact_mode,
        friction_lower=(0.05, 0.0, 0.0),
        friction_upper=(5.0, 0.2, 0.05),
        initial_friction=friction,
        apply_c2r=apply_c2r,
        no_support_plane=args.no_support_plane,
        support_plane_z=args.support_plane_z,
        no_save=True,
    )


def reset_data(scene, data: mujoco.MjData) -> None:
    data.qpos[:] = scene.initial_qpos
    data.qvel[:] = 0.0
    data.act[:] = 0.0
    data.ctrl[:] = scene.initial_ctrl
    mujoco.mj_forward(scene.model, data)


def playback(scene, sequence, speed: float, loop: bool) -> None:
    data = mujoco.MjData(scene.model)
    reset_data(scene, data)

    sim_steps_per_frame = max(1, int(round(sequence.dt / scene.model.opt.timestep)))
    wall_dt = sequence.dt / max(speed, 1e-6)

    print(
        "playing fitted rollout "
        f"frames={sequence.frame_ids[0]}..{sequence.frame_ids[-1]} "
        f"n={len(sequence.frame_ids)} friction={scene.model.geom_friction[scene.object_geom_id]}"
    )
    print("close the MuJoCo window to stop")

    with mujoco.viewer.launch_passive(scene.model, data) as viewer:
        while viewer.is_running():
            reset_data(scene, data)

            viewer.sync()
            print(
                f"\rframe={int(sequence.frame_ids[0])} "
                f"book_pos={data.xpos[scene.object_body_id]}",
                end="",
                flush=True,
            )
            time.sleep(wall_dt)

            for ctrl, frame_id in zip(sequence.controls[1:], sequence.frame_ids[1:]):
                if not viewer.is_running():
                    break

                data.ctrl[:] = ctrl
                frame_start = time.time()
                for _ in range(sim_steps_per_frame):
                    mujoco.mj_step(scene.model, data)

                viewer.sync()
                print(
                    f"\rframe={int(frame_id)} "
                    f"book_pos={data.xpos[scene.object_body_id]}",
                    end="",
                    flush=True,
                )

                sleep_time = wall_dt - (time.time() - frame_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            print()
            if not loop:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.03)
                break


def main() -> None:
    args = parse_args()
    result = np.load(args.result)
    friction = np.asarray(result["friction"], dtype=np.float64)
    frame_ids = np.asarray(result["frame_ids"], dtype=np.int64)
    result_apply_c2r = bool(result["apply_c2r"]) if "apply_c2r" in result.files else True
    apply_c2r = result_apply_c2r if args.apply_c2r is None else args.apply_c2r
    print(f"apply_c2r={apply_c2r}")

    scene_args = make_scene_args(args, friction, frame_ids, apply_c2r)
    scene, sequence = build_scene(scene_args)

    expected = np.asarray(frame_ids)
    actual = np.asarray(sequence.frame_ids)
    if len(expected) != len(actual) or not np.array_equal(expected, actual):
        print(
            "warning: replay frame_ids do not exactly match result file; "
            f"result={expected[:3]}..{expected[-3:]} replay={actual[:3]}..{actual[-3:]}"
        )

    playback(scene, sequence, args.speed, args.loop)


if __name__ == "__main__":
    main()
