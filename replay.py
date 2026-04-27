"""Kinematic replay for robot qpos and tracked object placement.

The script directly writes synchronized recorded arm/hand motion into robot
qpos and the tracked object trajectory into the object's freejoint qpos, then
calls ``mj_forward``. It does not step physics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from util.load_data import resample_to


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/home/capture15/shared_data/capture/eccv2026/inspire_f1")
DEFAULT_MESH_ROOT = Path("/home/capture15/shared_data/mesh_blender")
DEFAULT_ROBOT_XML = ROOT / "rsc/robot/xarm_inspire_f1.mjcf"

OBJECT_BODY = "tracked_object"
OBJECT_GEOM = "tracked_object_geom"
OBJECT_FREEJOINT = "tracked_object_freejoint"
SUPPORT_PLANE_GEOM = "support_plane"

ARM_ACTUATORS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
HAND_ACTUATORS = (
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
)
COUPLED_JOINTS = {
    "right_thumb_3_joint": ("right_thumb_2_joint", 1.2953),
    "right_thumb_4_joint": ("right_thumb_3_joint", 0.8962),
    "right_index_2_joint": ("right_index_1_joint", 1.1545),
    "right_middle_2_joint": ("right_middle_1_joint", 1.1545),
    "right_ring_2_joint": ("right_ring_1_joint", 1.1545),
    "right_little_2_joint": ("right_little_1_joint", 1.1545),
}
@dataclass(frozen=True)
class SequenceData:
    controls: np.ndarray
    object_qpos: np.ndarray
    object_pos: np.ndarray
    object_quat: np.ndarray
    frame_ids: np.ndarray
    dt: float


@dataclass(frozen=True)
class SceneInfo:
    model: mujoco.MjModel
    object_body_id: int
    object_geom_id: int
    object_qposadr: int
    joint_qposadr: dict[str, int]
    initial_qpos: np.ndarray
    initial_ctrl: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("object_name", nargs="?", default="apple")
    parser.add_argument("episode_number", nargs="?", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sequence-dir", type=Path, default=None)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--object-mesh", type=Path, default=None)
    parser.add_argument("--object-pose-npz", type=Path, default=None)
    parser.add_argument("--pose-dir", type=Path, default=None)
    parser.add_argument("--arm-file", default="position.npy")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--mode", choices=("kinematic",), default="kinematic")
    parser.add_argument("--hand-command-min", type=float, default=1000.0)
    parser.add_argument("--hand-command-max", type=float, default=2000.0)
    parser.add_argument("--apply-c2r", dest="apply_c2r", action="store_true", default=True)
    parser.add_argument("--no-apply-c2r", dest="apply_c2r", action="store_false")
    parser.add_argument("--direct-c2r", action="store_true", default=True)
    parser.add_argument("--inverse-c2r", dest="direct_c2r", action="store_false")
    parser.add_argument("--support-plane", action="store_true")
    parser.add_argument("--support-plane-z", type=float, default=None)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--print-camera-pose", action="store_true")
    return parser.parse_args()


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )
    return quat_wxyz / np.linalg.norm(quat_wxyz)


def wxyz_to_rotation(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def resolve_sequence_dir(args: argparse.Namespace) -> Path:
    if args.sequence_dir is not None:
        return args.sequence_dir.resolve()
    return (args.dataset_root / args.object_name / str(args.episode_number)).resolve()


def resolve_object_mesh(args: argparse.Namespace) -> Path:
    if args.object_mesh is not None:
        return args.object_mesh.resolve()

    candidates = [
        DEFAULT_MESH_ROOT / args.object_name / f"{args.object_name}.obj",
        DEFAULT_MESH_ROOT / args.object_name / f"{args.object_name}_remeshed.obj",
        ROOT / "rsc/object" / args.object_name / f"{args.object_name}.obj",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find an object mesh. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_object_pose_npz(args: argparse.Namespace, sequence_dir: Path) -> Path | None:
    if args.object_pose_npz is not None:
        return args.object_pose_npz.resolve()
    return None


def resolve_pose_dir(args: argparse.Namespace, sequence_dir: Path) -> Path | None:
    if args.pose_dir is not None:
        return args.pose_dir.resolve()

    refined_pose_dir = sequence_dir / "sequence_refine_output/refined_world_poses"
    if refined_pose_dir.is_dir():
        return refined_pose_dir.resolve()

    track_root = sequence_dir / "object_tracking_result"
    if not track_root.exists():
        return None

    pose_dirs = sorted(path for path in track_root.glob("*/poses") if path.is_dir())
    return pose_dirs[0].resolve() if pose_dirs else None


def load_series_with_timestamps(
    data_dir: Path, candidates: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    for name in candidates:
        data_path = data_dir / name
        if not data_path.exists():
            continue

        data = np.load(data_path, allow_pickle=True)
        stem_time = data_dir / f"{data_path.stem}_time.npy"
        if stem_time.exists():
            times = np.load(stem_time, allow_pickle=True)
        elif (data_dir / "time.npy").exists():
            times = np.load(data_dir / "time.npy", allow_pickle=True)
        else:
            times = np.arange(data.shape[0], dtype=float)

        data = np.asarray(data, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if len(times) != data.shape[0]:
            n = min(len(times), data.shape[0])
            data = data[:n]
            times = times[:n]
        return data, times

    raise FileNotFoundError(f"No data found in {data_dir} for {candidates}")


def load_video_timeline(sequence_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    frame_path = sequence_dir / "raw/timestamps/frame_id.npy"
    time_path = sequence_dir / "raw/timestamps/timestamp.npy"
    if not frame_path.exists() or not time_path.exists():
        return None

    frame_ids = np.asarray(np.load(frame_path, allow_pickle=True), dtype=np.int64).reshape(-1)
    times = np.asarray(np.load(time_path, allow_pickle=True), dtype=np.float64).reshape(-1)
    n = min(len(frame_ids), len(times))
    return frame_ids[:n], times[:n]


def object_times_from_frames(
    sequence_dir: Path,
    object_frame_ids: np.ndarray,
    fallback_start: float,
    fallback_end: float,
) -> np.ndarray:
    timeline = load_video_timeline(sequence_dir)
    if timeline is None:
        return np.linspace(fallback_start, fallback_end, len(object_frame_ids))

    video_frame_ids, video_times = timeline
    if (
        len(object_frame_ids) <= len(video_times)
        and np.array_equal(object_frame_ids, np.arange(len(object_frame_ids)))
    ):
        return video_times[: len(object_frame_ids)]

    by_frame = {int(frame): float(t) for frame, t in zip(video_frame_ids, video_times)}
    object_times = np.array(
        [by_frame.get(int(frame), np.nan) for frame in object_frame_ids], dtype=np.float64
    )
    missing = np.isnan(object_times)
    if np.all(missing):
        return np.linspace(video_times[0], video_times[-1], len(object_frame_ids))
    if np.any(missing):
        valid_index = np.flatnonzero(~missing)
        object_times[missing] = np.interp(
            np.flatnonzero(missing), valid_index, object_times[valid_index]
        )
    return object_times


def transforms_to_qpos(transforms: np.ndarray) -> np.ndarray:
    object_qpos = np.zeros((len(transforms), 7), dtype=np.float64)
    for i, transform in enumerate(transforms):
        object_qpos[i, :3] = transform[:3, 3]
        object_qpos[i, 3:] = matrix_to_wxyz(transform)
    return object_qpos


def load_object_trajectory(
    sequence_dir: Path,
    npz_path: Path | None,
    pose_dir: Path | None,
    apply_c2r: bool,
    direct_c2r: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if pose_dir is not None:
        pose_files = sorted(pose_dir.glob("pose_*.txt"))
        if not pose_files:
            raise FileNotFoundError(f"No pose_*.txt files found in {pose_dir}")
        frame_ids = np.array(
            [int(path.stem.split("_")[-1]) for path in pose_files], dtype=np.int64
        )
        transforms = np.stack([np.loadtxt(path, dtype=np.float64) for path in pose_files])
    elif npz_path is not None and npz_path.exists():
        npz = np.load(npz_path, allow_pickle=True)
        keys = sorted(npz.files, key=lambda key: int(key.split("_")[-1]))
        frame_ids = np.array([int(key.split("_")[-1]) for key in keys], dtype=np.int64)
        transforms = np.stack([np.asarray(npz[key], dtype=np.float64) for key in keys])
    else:
        raise FileNotFoundError(
            "No object trajectory found. Expected pose txt files under "
            f"{sequence_dir / 'sequence_refine_output/refined_world_poses'}"
        )

    if transforms.shape[1:] != (4, 4):
        raise ValueError(f"Expected object poses shaped Nx4x4, got {transforms.shape}")
    if apply_c2r:
        c2r_path = sequence_dir / "C2R.npy"
        if not c2r_path.exists():
            raise FileNotFoundError(f"--apply-c2r requested, but {c2r_path} is missing")
        c2r = np.load(c2r_path)
        if direct_c2r:
            transforms = np.einsum("ij,njk->nik", c2r, transforms)
        else:
            transforms = np.einsum("ij,njk->nik", np.linalg.inv(c2r), transforms)

    return frame_ids, transforms_to_qpos(transforms)


def command_to_hand_ctrl(
    commands: np.ndarray, ctrlrange: np.ndarray, command_min: float, command_max: float
) -> np.ndarray:
    scale = (commands - command_min) / (command_max - command_min)
    scale = np.clip(scale, 0.0, 1.0)
    return ctrlrange[:, 0] + scale * (ctrlrange[:, 1] - ctrlrange[:, 0])


def load_sequence(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    sequence_dir: Path,
    object_pose_npz: Path | None,
    pose_dir: Path | None,
) -> SequenceData:
    raw_dir = sequence_dir / "raw"
    arm_candidates = (args.arm_file, "position.npy", "action_qpos.npy", "action.npy")
    arm_qpos, arm_times = load_series_with_timestamps(raw_dir / "arm", arm_candidates)
    hand_commands, hand_times = load_series_with_timestamps(
        raw_dir / "hand", ("right_joint_states.npy", "right_commands.npy")
    )

    frame_ids, object_qpos = load_object_trajectory(
        sequence_dir,
        object_pose_npz,
        pose_dir,
        args.apply_c2r,
        args.direct_c2r,
    )
    object_times = object_times_from_frames(
        sequence_dir,
        frame_ids,
        float(arm_times[0]),
        float(arm_times[-1]),
    )

    arm_sync = resample_to(arm_times, np.asarray(arm_qpos, dtype=np.float64), object_times)
    hand_sync = resample_to(hand_times, hand_commands, object_times)

    if np.nanmax(np.abs(arm_sync)) > 2.0 * np.pi:
        arm_sync = np.deg2rad(arm_sync)

    ctrlrange = model.actuator_ctrlrange.copy()
    arm_ctrl = np.clip(arm_sync, ctrlrange[:6, 0], ctrlrange[:6, 1])
    hand_ctrl = command_to_hand_ctrl(
        hand_sync, ctrlrange[6:12], args.hand_command_min, args.hand_command_max
    )
    controls = np.concatenate([arm_ctrl, hand_ctrl], axis=1)

    keep = np.arange(len(object_qpos))
    if args.start_frame is not None:
        keep = keep[frame_ids >= args.start_frame]
    if args.stride > 1:
        keep = keep[:: args.stride]
    if args.max_frames is not None:
        keep = keep[: args.max_frames]
    if len(keep) < 2:
        raise ValueError("Need at least two object frames after stride/max-frame filtering")

    object_times = object_times[keep]
    dt = float(np.mean(np.diff(object_times)))
    object_qpos = object_qpos[keep]

    return SequenceData(
        controls=controls[keep],
        object_qpos=object_qpos,
        object_pos=object_qpos[:, :3],
        object_quat=object_qpos[:, 3:],
        frame_ids=frame_ids[keep],
        dt=dt,
    )


def read_obj_vertices(mesh_path: Path) -> np.ndarray:
    vertices = []
    with mesh_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(item) for item in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices found in {mesh_path}")
    return np.asarray(vertices, dtype=np.float64)


def infer_support_plane_z(object_mesh: Path, object_qpos: np.ndarray) -> float:
    vertices = read_obj_vertices(object_mesh)
    rotation = wxyz_to_rotation(object_qpos[3:])
    world_vertices = vertices @ rotation.T + object_qpos[:3]
    return float(np.min(world_vertices[:, 2]) - 0.002)


def build_combined_model(
    robot_xml: Path,
    object_mesh: Path,
    initial_object_qpos: np.ndarray,
    add_support_plane: bool,
    support_plane_z: float | None,
) -> mujoco.MjModel:
    root = ET.parse(robot_xml).getroot()

    for mesh_node in root.find("asset").findall("mesh"):
        mesh_file = mesh_node.attrib.get("file")
        if mesh_file and not Path(mesh_file).is_absolute():
            mesh_node.set("file", str((robot_xml.parent / mesh_file).resolve()))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No worldbody found in {robot_xml}")

    ET.SubElement(root.find("asset"), "mesh", name="tracked_object_mesh", file=str(object_mesh))

    if add_support_plane:
        plane_z = (
            support_plane_z
            if support_plane_z is not None
            else infer_support_plane_z(object_mesh, initial_object_qpos)
        )
        ET.SubElement(
            worldbody,
            "geom",
            name=SUPPORT_PLANE_GEOM,
            type="plane",
            pos=f"0 0 {plane_z}",
            size="0.8 0.8 0.02",
            friction="1.0 0.005 0.0001",
            contype="0",
            conaffinity="0",
            rgba="0.35 0.35 0.35 0.25",
        )

    object_body = ET.SubElement(
        worldbody,
        "body",
        name=OBJECT_BODY,
        pos=" ".join(map(str, initial_object_qpos[:3])),
        quat=" ".join(map(str, initial_object_qpos[3:])),
    )
    ET.SubElement(object_body, "freejoint", name=OBJECT_FREEJOINT)
    ET.SubElement(
        object_body,
        "geom",
        name=OBJECT_GEOM,
        type="mesh",
        mesh="tracked_object_mesh",
        mass="0.2",
        contype="0",
        conaffinity="0",
        condim="3",
        rgba="0.78 0.65 0.42 1",
    )

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model


def name_to_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type} named {name}")
    return obj_id


def make_initial_state(
    model: mujoco.MjModel, first_ctrl: np.ndarray, first_object_qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    joint_qposadr = {}
    qpos = np.zeros(model.nq, dtype=np.float64)

    for joint_name in ARM_ACTUATORS + HAND_ACTUATORS:
        joint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qposadr = model.jnt_qposadr[joint_id]
        actuator_id = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        qpos[qposadr] = first_ctrl[actuator_id]
        joint_qposadr[joint_name] = qposadr

    for joint_name, (parent_name, gain) in COUPLED_JOINTS.items():
        joint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qposadr = model.jnt_qposadr[joint_id]
        qpos[qposadr] = gain * qpos[joint_qposadr[parent_name]]
        joint_qposadr[joint_name] = qposadr

    freejoint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_FREEJOINT)
    object_qposadr = model.jnt_qposadr[freejoint_id]
    qpos[object_qposadr : object_qposadr + 7] = first_object_qpos

    ctrl = np.clip(first_ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    return qpos, ctrl, joint_qposadr


def build_scene(args: argparse.Namespace) -> tuple[SceneInfo, SequenceData]:
    sequence_dir = resolve_sequence_dir(args)
    object_mesh = resolve_object_mesh(args)
    object_pose_npz = resolve_object_pose_npz(args, sequence_dir)
    pose_dir = resolve_pose_dir(args, sequence_dir)
    _, all_object_qpos = load_object_trajectory(
        sequence_dir,
        object_pose_npz,
        pose_dir,
        args.apply_c2r,
        args.direct_c2r,
    )
    model = build_combined_model(
        args.robot_xml.resolve(),
        object_mesh,
        all_object_qpos[0],
        args.support_plane,
        args.support_plane_z,
    )
    sequence = load_sequence(args, model, sequence_dir, object_pose_npz, pose_dir)

    if args.support_plane and args.support_plane_z is None:
        support_plane_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, SUPPORT_PLANE_GEOM)
        model.geom_pos[support_plane_id, 2] = infer_support_plane_z(
            object_mesh, sequence.object_qpos[0]
        )

    initial_qpos, initial_ctrl, joint_qposadr = make_initial_state(
        model, sequence.controls[0], sequence.object_qpos[0]
    )
    object_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    object_geom_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
    freejoint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_FREEJOINT)

    return (
        SceneInfo(
            model=model,
            object_body_id=object_body_id,
            object_geom_id=object_geom_id,
            object_qposadr=model.jnt_qposadr[freejoint_id],
            joint_qposadr=joint_qposadr,
            initial_qpos=initial_qpos,
            initial_ctrl=initial_ctrl,
        ),
        sequence,
    )


def print_camera_to_robot_check(args: argparse.Namespace, frame_id: int) -> None:
    sequence_dir = resolve_sequence_dir(args)
    pose_dir = resolve_pose_dir(args, sequence_dir)
    if pose_dir is None:
        print("No pose_*.txt directory found for camera pose check")
        return
    c2r_path = sequence_dir / "C2R.npy"
    pose_path = pose_dir / f"pose_{int(frame_id):06d}.txt"
    if not pose_path.exists():
        print(f"{pose_path} missing")
        return
    camera_object = np.loadtxt(pose_path, dtype=np.float64)
    c2r = np.load(c2r_path)
    direct_object = c2r @ camera_object
    visualizer_object = np.linalg.inv(c2r) @ camera_object
    print("camera/world object pose check")
    print(f"  visualize_all.py formula: T_robot_object = inv(C2R) @ T_capture_object")
    print(f"  pose file: {pose_path}")
    print(f"  camera object xyz: {np.array2string(camera_object[:3, 3], precision=5)}")
    print(f"  inv(C2R) @ camera object xyz: {np.array2string(visualizer_object[:3, 3], precision=5)}")
    print(f"  C2R @ camera object xyz: {np.array2string(direct_object[:3, 3], precision=5)}")


def set_robot_qpos_from_ctrl(scene, data: mujoco.MjData, ctrl: np.ndarray) -> None:
    joint_names = ARM_ACTUATORS + HAND_ACTUATORS
    for actuator_index, joint_name in enumerate(joint_names):
        data.qpos[scene.joint_qposadr[joint_name]] = ctrl[actuator_index]

    for joint_name, (parent_name, gain) in COUPLED_JOINTS.items():
        data.qpos[scene.joint_qposadr[joint_name]] = (
            gain * data.qpos[scene.joint_qposadr[parent_name]]
        )


def set_kinematic_frame(scene, sequence, data: mujoco.MjData, index: int) -> None:
    data.qpos[:] = scene.initial_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = sequence.controls[index]
    set_robot_qpos_from_ctrl(scene, data, sequence.controls[index])
    data.qpos[scene.object_qposadr : scene.object_qposadr + 7] = sequence.object_qpos[index]
    mujoco.mj_forward(scene.model, data)


def print_debug_frame(
    scene,
    sequence,
    data: mujoco.MjData,
    index: int,
    prefix: str,
    tracked_object: bool,
) -> None:
    arm = sequence.controls[index, :6]
    hand = sequence.controls[index, 6:12]
    obj_qpos = sequence.object_qpos[index]
    sim_obj_pos = data.xpos[scene.object_body_id]
    obj_label = "tracked_obj_pos" if tracked_object else "sim_obj_pos"
    print(
        f"{prefix} frame={int(sequence.frame_ids[index])} "
        f"arm_qpos(rad)={np.array2string(arm, precision=3)} "
        f"hand_ctrl={np.array2string(hand, precision=3)} "
        f"gt_obj_pos={np.array2string(obj_qpos[:3], precision=4)} "
        f"{obj_label}={np.array2string(sim_obj_pos, precision=4)}"
    )


def run_headless(scene, sequence, mode: str, print_every: int) -> None:
    data = mujoco.MjData(scene.model)
    for i in range(len(sequence.frame_ids)):
        set_kinematic_frame(scene, sequence, data, i)
        if i == 0 or i == len(sequence.frame_ids) - 1 or i % print_every == 0:
            print_debug_frame(scene, sequence, data, i, "kinematic", True)


def play_kinematic(scene, sequence, args: argparse.Namespace) -> None:
    data = mujoco.MjData(scene.model)
    wall_dt = sequence.dt / max(args.speed, 1e-6)

    with mujoco.viewer.launch_passive(scene.model, data) as viewer:
        while viewer.is_running():
            for i in range(len(sequence.frame_ids)):
                if not viewer.is_running():
                    break
                frame_start = time.time()
                set_kinematic_frame(scene, sequence, data, i)
                viewer.sync()
                if i == 0 or i == len(sequence.frame_ids) - 1 or i % args.print_every == 0:
                    print_debug_frame(scene, sequence, data, i, "kinematic", True)
                sleep_time = wall_dt - (time.time() - frame_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if not args.loop:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.03)
                break


def main() -> None:
    args = parse_args()
    scene, sequence = build_scene(args)
    sequence_dir = resolve_sequence_dir(args)
    pose_dir = resolve_pose_dir(args, sequence_dir)
    object_pose_npz = resolve_object_pose_npz(args, sequence_dir)
    object_source = pose_dir if pose_dir is not None else object_pose_npz

    print(
        f"object={args.object_name} episode={args.episode_number} "
        f"sequence_dir={sequence_dir} "
        f"object_trajectory={object_source} "
        f"mode=kinematic apply_c2r={args.apply_c2r} "
        f"direct_c2r={args.direct_c2r} "
        f"frames={int(sequence.frame_ids[0])}..{int(sequence.frame_ids[-1])} "
        f"n={len(sequence.frame_ids)} dt={sequence.dt:.4f}s"
    )
    if args.apply_c2r and args.direct_c2r:
        print("robot is fixed in the MuJoCo world; object pose is C2R @ object_seq[t]")
    elif args.apply_c2r:
        print("robot is fixed in the MuJoCo world; object pose is inv(C2R) @ object_seq[t]")
    else:
        print("robot is fixed in the MuJoCo world; object pose is raw object_seq[t]")
    print(f"first object qpos={np.array2string(sequence.object_qpos[0], precision=5)}")
    print(f"first robot ctrl={np.array2string(sequence.controls[0], precision=5)}")
    if args.print_camera_pose:
        print_camera_to_robot_check(args, int(sequence.frame_ids[0]))

    if args.headless:
        run_headless(scene, sequence, args.mode, args.print_every)
    else:
        play_kinematic(scene, sequence, args)


if __name__ == "__main__":
    main()
