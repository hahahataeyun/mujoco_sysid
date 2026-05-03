"""Replay robot qpos with tracked, dynamic, or side-by-side object placement.

Kinematic mode writes synchronized recorded arm/hand motion and tracked object
poses directly into qpos. Dynamic mode writes the object pose only at the first
timestep, then lets MuJoCo contact dynamics move the object while the robot is
driven by MJCF PD actuators. Both mode shows synchronized kinematic and dynamic
replay windows.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from util.load_data import resample_to


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/home/capture15/shared_data/capture/eccv2026/inspire_f1")
DEFAULT_ROBOT_XML = ROOT / "robot/xarm_inspire_f1.mjcf"
DEFAULT_ROBOT_OVERRIDE_YAML = ROOT / "robot/replay_overrides.yaml"
ROBOT_XML_DIRS = (ROOT / "robot", ROOT / "rsc/robot", ROOT / "pd_id/initial_mjcf")
OBJECT_XML_DIR = ROOT / "rsc/object"

OBJECT_BODY = "tracked_object"
OBJECT_FREEJOINT = "tracked_object_freejoint"
SUPPORT_PLANE_GEOM = "support_plane"
KINEMATIC_SUPPORT_PLANE_OFFSET = -0.002
DYNAMIC_SUPPORT_PLANE_OFFSET = 0.0
BOTH_KINEMATIC_OFFSET = np.array([0.0, -0.6, 0.0], dtype=np.float64)
BOTH_DYNAMIC_OFFSET = np.array([0.0, 0.6, 0.0], dtype=np.float64)
RESTART_KEY = ord("R")

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
    robot_qpos: np.ndarray
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
    object_qposadr: int
    joint_qposadr: dict[str, int]
    actuator_ids: np.ndarray
    scene_offset: np.ndarray
    initial_qpos: np.ndarray
    initial_ctrl: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("object_name", nargs="?", default="apple")
    parser.add_argument("episode_number", nargs="?", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sequence-dir", type=Path, default=None)
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=DEFAULT_ROBOT_XML,
        help="Robot MJCF path, or a model name under robot, rsc/robot, or pd_id/initial_mjcf.",
    )
    parser.add_argument(
        "--robot-override-yaml",
        type=Path,
        default=DEFAULT_ROBOT_OVERRIDE_YAML,
        help="YAML file with robot MJCF default-class overrides.",
    )
    parser.add_argument(
        "--object-xml",
        type=Path,
        default=None,
        help="Object MJCF path. Defaults to rsc/object/<object>/<object>.mjcf.",
    )
    parser.add_argument("--object-pose-npz", type=Path, default=None)
    parser.add_argument("--pose-dir", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--mode", choices=("kinematic", "dynamic", "both"), default="kinematic")
    parser.add_argument("--support-plane", action="store_true")
    parser.add_argument("--support-plane-z", type=float, default=None)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--visualize-contacts", action="store_true")
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


def resolve_robot_xml(args: argparse.Namespace) -> Path:
    robot_xml = args.robot_xml
    if robot_xml.parent != Path(".") or robot_xml.is_absolute():
        return robot_xml.resolve()

    names = [robot_xml]
    if robot_xml.suffix == "":
        names.append(robot_xml.with_suffix(".mjcf"))

    for name in names:
        if name.exists():
            return name.resolve()
        for directory in ROBOT_XML_DIRS:
            candidate = directory / name
            if candidate.exists():
                return candidate.resolve()

    return names[-1].resolve()


def resolve_robot_override_yaml(args: argparse.Namespace) -> Path | None:
    if args.robot_override_yaml is None:
        return None
    path = args.robot_override_yaml.resolve()
    return path if path.exists() else None


def resolve_object_xml(args: argparse.Namespace) -> Path:
    if args.object_xml is not None:
        return args.object_xml.resolve()

    candidates = [
        OBJECT_XML_DIR / args.object_name / f"{args.object_name}.mjcf",
        OBJECT_XML_DIR / args.object_name / f"{args.object_name}.xml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find an object MJCF. Tried: "
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
    c2r_path = sequence_dir / "C2R.npy"
    if not c2r_path.exists():
        raise FileNotFoundError(f"Missing camera-to-robot transform: {c2r_path}")
    c2r = np.load(c2r_path)
    transforms = np.einsum("ij,njk->nik", np.linalg.inv(c2r), transforms)

    return frame_ids, transforms_to_qpos(transforms)


def inspire_f1_raw_to_qpos(action: np.ndarray) -> np.ndarray:
    """Convert Inspire F1 raw hand action to MJCF actuator order in radians.

    Raw action order is:
      little, ring, middle, index, thumb_2, thumb_1

    MJCF actuator order is:
      thumb_1, thumb_2, index, middle, ring, little
    """
    action = np.asarray(action, dtype=np.float64)
    qpos_raw_order = np.zeros_like(action, dtype=np.float64)
    qpos_raw_order[:, 0] = (1800.0 - action[:, 0]) * np.pi / 1800.0
    qpos_raw_order[:, 1] = (1350.0 - action[:, 1]) * np.pi / 1800.0
    qpos_raw_order[:, 2] = (1740.0 - action[:, 2]) * np.pi / 1800.0
    qpos_raw_order[:, 3] = (1740.0 - action[:, 3]) * np.pi / 1800.0
    qpos_raw_order[:, 4] = (1740.0 - action[:, 4]) * np.pi / 1800.0
    qpos_raw_order[:, 5] = (1740.0 - action[:, 5]) * np.pi / 1800.0

    return qpos_raw_order


def load_sequence(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    sequence_dir: Path,
    object_pose_npz: Path | None,
    pose_dir: Path | None,
) -> SequenceData:
    raw_dir = sequence_dir / "raw"
    arm_qpos, arm_times = load_series_with_timestamps(raw_dir / "arm", ("position.npy",))
    hand_qpos_raw, hand_qpos_times = load_series_with_timestamps(
        raw_dir / "hand", ("right_joint_states.npy",)
    )
    hand_commands_raw, hand_command_times = load_series_with_timestamps(
        raw_dir / "hand", ("right_commands.npy",)
    )

    frame_ids, object_qpos = load_object_trajectory(
        sequence_dir,
        object_pose_npz,
        pose_dir,
    )
    object_times = object_times_from_frames(
        sequence_dir,
        frame_ids,
        float(arm_times[0]),
        float(arm_times[-1]),
    )

    arm_sync = resample_to(arm_times, np.asarray(arm_qpos, dtype=np.float64), object_times)
    hand_qpos_sync = resample_to(hand_qpos_times, hand_qpos_raw, object_times)
    hand_command_sync = resample_to(hand_command_times, hand_commands_raw, object_times)

    if np.nanmax(np.abs(arm_sync)) > 2.0 * np.pi:
        arm_sync = np.deg2rad(arm_sync)

    ctrlrange = model.actuator_ctrlrange.copy()
    arm_ctrl = np.clip(arm_sync, ctrlrange[:6, 0], ctrlrange[:6, 1])
    hand_qpos = inspire_f1_raw_to_qpos(hand_qpos_sync)
    hand_command_qpos = inspire_f1_raw_to_qpos(hand_command_sync)
    hand_ctrl = np.clip(hand_command_qpos, ctrlrange[6:12, 0], ctrlrange[6:12, 1])
    robot_qpos = np.concatenate([arm_sync, hand_qpos], axis=1)
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
        robot_qpos=robot_qpos[keep],
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


def mesh_asset_name(mesh: ET.Element) -> str | None:
    name = mesh.attrib.get("name")
    if name:
        return name
    file_name = mesh.attrib.get("file")
    return Path(file_name).stem if file_name else None


def resolve_xml_file_paths(root: ET.Element, xml_dir: Path) -> None:
    fallback_dirs = []
    for file_node in root.findall(".//*[@file]"):
        file_name = file_node.attrib.get("file")
        if file_name and Path(file_name).is_absolute():
            fallback_dirs.append(Path(file_name).parent)

    for file_node in root.findall(".//*[@file]"):
        file_name = file_node.attrib.get("file")
        if not file_name or Path(file_name).is_absolute():
            continue

        candidate = xml_dir / file_name
        if not candidate.exists():
            for fallback_dir in fallback_dirs:
                fallback_candidate = fallback_dir / file_name
                if fallback_candidate.exists():
                    candidate = fallback_candidate
                    break
        file_node.set("file", str(candidate.resolve()))


def load_object_xml(object_xml: Path) -> ET.Element:
    root = ET.parse(object_xml).getroot()
    resolve_xml_file_paths(root, object_xml.parent)
    return root


def iter_object_mesh_files(object_root: ET.Element) -> list[Path]:
    mesh_files = []
    for mesh in object_root.findall("./asset/mesh"):
        file_name = mesh.attrib.get("file")
        if file_name and file_name.endswith(".obj"):
            mesh_files.append(Path(file_name))
    if not mesh_files:
        raise ValueError("Object MJCF does not contain any OBJ mesh assets")
    return mesh_files


def infer_support_plane_z_from_object_xml(
    object_xml: Path, object_qpos: np.ndarray, plane_offset: float
) -> float:
    object_root = load_object_xml(object_xml)
    vertices = np.concatenate(
        [read_obj_vertices(mesh_path) for mesh_path in iter_object_mesh_files(object_root)],
        axis=0,
    )
    rotation = wxyz_to_rotation(object_qpos[3:])
    world_vertices = vertices @ rotation.T + object_qpos[:3]
    return float(np.min(world_vertices[:, 2]) + plane_offset)


def merge_default_classes(target_root: ET.Element, source_root: ET.Element, prefix: str) -> None:
    source_default = source_root.find("default")
    if source_default is None:
        return

    target_default = target_root.find("default")
    if target_default is None:
        target_default = ET.SubElement(target_root, "default")

    for child in source_default:
        child_copy = copy.deepcopy(child)
        class_name = child_copy.attrib.get("class")
        if class_name:
            child_copy.set("class", f"{prefix}{class_name}")
        target_default.append(child_copy)


def prefix_object_element(
    element: ET.Element,
    prefix: str,
    asset_names: set[str],
    default_classes: set[str],
) -> None:
    for node in element.iter():
        name = node.attrib.get("name")
        if name:
            node.set("name", f"{prefix}{name}")

        mesh = node.attrib.get("mesh")
        if mesh in asset_names:
            node.set("mesh", f"{prefix}{mesh}")

        material = node.attrib.get("material")
        if material in asset_names:
            node.set("material", f"{prefix}{material}")

        texture = node.attrib.get("texture")
        if texture in asset_names:
            node.set("texture", f"{prefix}{texture}")

        class_name = node.attrib.get("class")
        if class_name in default_classes:
            node.set("class", f"{prefix}{class_name}")


def append_object_assets(
    target_asset: ET.Element,
    object_root: ET.Element,
    prefix: str,
) -> set[str]:
    source_asset = object_root.find("asset")
    if source_asset is None:
        return set()

    asset_names = set()
    for asset in source_asset:
        if asset.tag == "mesh":
            name = mesh_asset_name(asset)
        else:
            name = asset.attrib.get("name")
        if name:
            asset_names.add(name)

    for asset in source_asset:
        asset_copy = copy.deepcopy(asset)
        if asset_copy.tag == "mesh" and "name" not in asset_copy.attrib:
            name = mesh_asset_name(asset_copy)
            if name:
                asset_copy.set("name", name)
        prefix_object_element(asset_copy, prefix, asset_names, set())
        target_asset.append(asset_copy)

    return asset_names


def object_default_classes(object_root: ET.Element) -> set[str]:
    source_default = object_root.find("default")
    if source_default is None:
        return set()
    return {
        child.attrib["class"]
        for child in source_default
        if "class" in child.attrib
    }


def find_object_body(object_root: ET.Element) -> ET.Element:
    worldbody = object_root.find("worldbody")
    if worldbody is None:
        raise ValueError("Object MJCF is missing worldbody")
    bodies = worldbody.findall("body")
    if len(bodies) != 1:
        raise ValueError(f"Expected exactly one top-level object body, got {len(bodies)}")
    return bodies[0]


def append_object_body(
    target_worldbody: ET.Element,
    object_root: ET.Element,
    *,
    prefix: str,
    initial_object_qpos: np.ndarray,
    offset: np.ndarray,
    physics_enabled: bool,
    asset_names: set[str],
) -> None:
    default_classes = object_default_classes(object_root)
    object_body = copy.deepcopy(find_object_body(object_root))
    prefix_object_element(object_body, prefix, asset_names, default_classes)
    if not physics_enabled:
        disable_contacts(object_body)

    object_qpos = object_qpos_with_offset(initial_object_qpos, offset)
    object_body.set("name", f"{prefix}{OBJECT_BODY}")
    object_body.set("pos", " ".join(map(str, object_qpos[:3])))
    object_body.set("quat", " ".join(map(str, object_qpos[3:])))

    freejoints = list(object_body.iter("freejoint"))
    if freejoints:
        freejoints[0].set("name", f"{prefix}{OBJECT_FREEJOINT}")
    else:
        object_body.insert(0, ET.Element("freejoint", name=f"{prefix}{OBJECT_FREEJOINT}"))

    target_worldbody.append(object_body)


def set_xml_attrs(element: ET.Element, attrs: dict) -> None:
    for key, value in attrs.items():
        element.set(str(key), str(value))


def apply_robot_xml_overrides(root: ET.Element, override_yaml: Path | None) -> None:
    if override_yaml is None:
        return

    with override_yaml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"Expected 'defaults' mapping in {override_yaml}")

    root_default = root.find("default")
    if root_default is None:
        root_default = ET.SubElement(root, "default")

    for class_name, class_config in defaults.items():
        if not isinstance(class_config, dict):
            raise ValueError(f"Expected mapping for default class {class_name} in {override_yaml}")

        class_default = root_default.find(f"default[@class='{class_name}']")
        if class_default is None:
            class_default = ET.SubElement(root_default, "default", {"class": str(class_name)})

        for child_tag, attrs in class_config.items():
            if not isinstance(attrs, dict):
                raise ValueError(
                    f"Expected attribute mapping for {class_name}.{child_tag} in {override_yaml}"
                )
            child = class_default.find(str(child_tag))
            if child is None:
                child = ET.SubElement(class_default, str(child_tag))
            set_xml_attrs(child, attrs)


def build_combined_model(
    robot_xml: Path,
    object_xml: Path,
    initial_object_qpos: np.ndarray,
    add_support_plane: bool,
    support_plane_z: float | None,
    physics_enabled: bool,
    robot_override_yaml: Path | None,
) -> mujoco.MjModel:
    root = ET.parse(robot_xml).getroot()
    apply_robot_xml_overrides(root, robot_override_yaml)
    object_root = load_object_xml(object_xml)
    merge_default_classes(root, object_root, "")
    option = root.find("option")
    if option is not None and physics_enabled:
        option.set("iterations", "100")

    resolve_xml_file_paths(root, robot_xml.parent)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No worldbody found in {robot_xml}")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    object_asset_names = append_object_assets(asset, object_root, "")

    if add_support_plane:
        plane_offset = (
            DYNAMIC_SUPPORT_PLANE_OFFSET
            if physics_enabled
            else KINEMATIC_SUPPORT_PLANE_OFFSET
        )
        plane_z = (
            support_plane_z
            if support_plane_z is not None
            else infer_support_plane_z_from_object_xml(
                object_xml, initial_object_qpos, plane_offset
            )
        )
        ET.SubElement(
            worldbody,
            "geom",
            name=SUPPORT_PLANE_GEOM,
            type="plane",
            pos=f"0 0 {plane_z}",
            size="0.8 0.8 0.02",
            friction="1.0 0.005 0.0001",
            condim="3",
            margin="0.001" if physics_enabled else "0",
            solref="0.001 1" if physics_enabled else "0.02 1",
            solimp="0.99 0.995 0.0001" if physics_enabled else "0.9 0.95 0.001",
            contype="1" if physics_enabled else "0",
            conaffinity="1" if physics_enabled else "0",
            rgba="0.35 0.35 0.35 0.25",
        )

    append_object_body(
        worldbody,
        object_root,
        prefix="",
        initial_object_qpos=initial_object_qpos,
        offset=np.zeros(3, dtype=np.float64),
        physics_enabled=physics_enabled,
        asset_names=object_asset_names,
    )

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model


def add_offset_to_pos(element: ET.Element, offset: np.ndarray) -> None:
    pos = np.fromstring(element.attrib.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
    if pos.size != 3:
        pos = np.zeros(3, dtype=np.float64)
    pos = pos + offset
    element.set("pos", " ".join(f"{value:.12g}" for value in pos))


def prefix_robot_element(element: ET.Element, prefix: str, asset_names: set[str]) -> None:
    for node in element.iter():
        name = node.attrib.get("name")
        if name:
            node.set("name", f"{prefix}{name}")

        mesh = node.attrib.get("mesh")
        if mesh in asset_names:
            node.set("mesh", f"{prefix}{mesh}")

        joint = node.attrib.get("joint")
        if joint:
            node.set("joint", f"{prefix}{joint}")

        for attr in ("joint1", "joint2"):
            joint_name = node.attrib.get(attr)
            if joint_name:
                node.set(attr, f"{prefix}{joint_name}")


def disable_contacts(element: ET.Element) -> None:
    for geom in element.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")


def append_prefixed_robot(
    target_root: ET.Element,
    robot_root: ET.Element,
    prefix: str,
    offset: np.ndarray,
    contacts_enabled: bool,
) -> None:
    target_asset = target_root.find("asset")
    target_worldbody = target_root.find("worldbody")
    target_actuator = target_root.find("actuator")
    target_equality = target_root.find("equality")
    if target_asset is None or target_worldbody is None or target_actuator is None:
        raise ValueError("Combined root is missing required sections")

    source_asset = robot_root.find("asset")
    asset_names = {
        asset.attrib["name"]
        for asset in source_asset.findall(".//*[@name]")
        if "name" in asset.attrib
    } if source_asset is not None else set()

    if source_asset is not None:
        for child in source_asset:
            child_copy = copy.deepcopy(child)
            prefix_robot_element(child_copy, prefix, asset_names)
            target_asset.append(child_copy)

    source_worldbody = robot_root.find("worldbody")
    if source_worldbody is None:
        raise ValueError("Robot XML is missing worldbody")
    for child in source_worldbody:
        child_copy = copy.deepcopy(child)
        prefix_robot_element(child_copy, prefix, asset_names)
        add_offset_to_pos(child_copy, offset)
        if not contacts_enabled:
            disable_contacts(child_copy)
        target_worldbody.append(child_copy)

    source_actuator = robot_root.find("actuator")
    if source_actuator is not None:
        for child in source_actuator:
            child_copy = copy.deepcopy(child)
            prefix_robot_element(child_copy, prefix, asset_names)
            target_actuator.append(child_copy)

    source_equality = robot_root.find("equality")
    if source_equality is not None:
        if target_equality is None:
            target_equality = ET.SubElement(target_root, "equality")
        for child in source_equality:
            child_copy = copy.deepcopy(child)
            prefix_robot_element(child_copy, prefix, asset_names)
            target_equality.append(child_copy)


def add_support_plane(
    worldbody: ET.Element,
    *,
    name: str,
    plane_z: float,
    offset: np.ndarray,
    physics_enabled: bool,
) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        name=name,
        type="plane",
        pos=f"{offset[0]} {offset[1]} {plane_z}",
        size="0.8 0.8 0.02",
        friction="1.0 0.005 0.0001",
        condim="3",
        margin="0.001" if physics_enabled else "0",
        solref="0.001 1" if physics_enabled else "0.02 1",
        solimp="0.99 0.995 0.0001" if physics_enabled else "0.9 0.95 0.001",
        contype="1" if physics_enabled else "0",
        conaffinity="1" if physics_enabled else "0",
        rgba="0.35 0.35 0.35 0.25",
    )


def build_both_model(
    robot_xml: Path,
    object_xml: Path,
    initial_object_qpos: np.ndarray,
    support_plane_z: float | None,
    robot_override_yaml: Path | None,
) -> mujoco.MjModel:
    robot_root = ET.parse(robot_xml).getroot()
    apply_robot_xml_overrides(robot_root, robot_override_yaml)
    resolve_xml_file_paths(robot_root, robot_xml.parent)
    object_root = load_object_xml(object_xml)

    root = ET.Element("mujoco", model="both_replay")
    for tag in ("compiler", "option", "default"):
        elem = robot_root.find(tag)
        if elem is not None:
            root.append(copy.deepcopy(elem))
    merge_default_classes(root, object_root, "kin_")
    merge_default_classes(root, object_root, "dyn_")
    option = root.find("option")
    if option is not None:
        option.set("iterations", "100")
    ET.SubElement(root, "asset")
    ET.SubElement(root, "worldbody")
    ET.SubElement(root, "actuator")
    ET.SubElement(root, "equality")

    append_prefixed_robot(root, robot_root, "kin_", BOTH_KINEMATIC_OFFSET, False)
    append_prefixed_robot(root, robot_root, "dyn_", BOTH_DYNAMIC_OFFSET, True)

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Combined root is missing asset or worldbody")
    kin_object_asset_names = append_object_assets(asset, object_root, "kin_")
    dyn_object_asset_names = append_object_assets(asset, object_root, "dyn_")

    plane_z = support_plane_z
    if plane_z is None:
        plane_z = infer_support_plane_z_from_object_xml(
            object_xml, initial_object_qpos, DYNAMIC_SUPPORT_PLANE_OFFSET
        )
    add_support_plane(
        worldbody,
        name=f"dyn_{SUPPORT_PLANE_GEOM}",
        plane_z=plane_z,
        offset=BOTH_DYNAMIC_OFFSET,
        physics_enabled=True,
    )
    append_object_body(
        worldbody,
        object_root,
        prefix="kin_",
        initial_object_qpos=initial_object_qpos,
        offset=BOTH_KINEMATIC_OFFSET,
        physics_enabled=False,
        asset_names=kin_object_asset_names,
    )
    append_object_body(
        worldbody,
        object_root,
        prefix="dyn_",
        initial_object_qpos=initial_object_qpos,
        offset=BOTH_DYNAMIC_OFFSET,
        physics_enabled=True,
        asset_names=dyn_object_asset_names,
    )

    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def name_to_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type} named {name}")
    return obj_id


def make_initial_state(
    model: mujoco.MjModel,
    first_robot_qpos: np.ndarray,
    first_ctrl: np.ndarray,
    first_object_qpos: np.ndarray,
    *,
    prefix: str = "",
    scene_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], np.ndarray]:
    joint_qposadr = {}
    actuator_ids = []
    qpos = np.zeros(model.nq, dtype=np.float64)
    offset = np.zeros(3, dtype=np.float64) if scene_offset is None else scene_offset

    for joint_name in ARM_ACTUATORS + HAND_ACTUATORS:
        joint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{joint_name}")
        actuator_id = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{joint_name}")
        joint_qposadr[joint_name] = model.jnt_qposadr[joint_id]
        actuator_ids.append(actuator_id)

    for joint_name, (parent_name, gain) in COUPLED_JOINTS.items():
        joint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{joint_name}")
        joint_qposadr[joint_name] = model.jnt_qposadr[joint_id]

    freejoint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{OBJECT_FREEJOINT}")
    object_qposadr = model.jnt_qposadr[freejoint_id]
    write_robot_qpos(qpos, joint_qposadr, first_robot_qpos)
    qpos[object_qposadr : object_qposadr + 7] = object_qpos_with_offset(
        first_object_qpos, offset
    )

    actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
    ctrl = np.zeros(model.nu, dtype=np.float64)
    ctrl[actuator_ids] = np.clip(
        first_ctrl,
        model.actuator_ctrlrange[actuator_ids, 0],
        model.actuator_ctrlrange[actuator_ids, 1],
    )
    return qpos, ctrl, joint_qposadr, actuator_ids


def make_scene_info(
    model: mujoco.MjModel,
    sequence: SequenceData,
    *,
    prefix: str = "",
    scene_offset: np.ndarray | None = None,
) -> SceneInfo:
    offset = np.zeros(3, dtype=np.float64) if scene_offset is None else scene_offset
    initial_qpos, initial_ctrl, joint_qposadr, actuator_ids = make_initial_state(
        model,
        sequence.robot_qpos[0],
        sequence.controls[0],
        sequence.object_qpos[0],
        prefix=prefix,
        scene_offset=offset,
    )
    object_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{OBJECT_BODY}")
    freejoint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{OBJECT_FREEJOINT}")

    return SceneInfo(
        model=model,
        object_body_id=object_body_id,
        object_qposadr=model.jnt_qposadr[freejoint_id],
        joint_qposadr=joint_qposadr,
        actuator_ids=actuator_ids,
        scene_offset=offset,
        initial_qpos=initial_qpos,
        initial_ctrl=initial_ctrl,
    )


def build_scene(args: argparse.Namespace) -> tuple[SceneInfo, SequenceData]:
    sequence_dir = resolve_sequence_dir(args)
    robot_xml = resolve_robot_xml(args)
    robot_override_yaml = resolve_robot_override_yaml(args)
    object_xml = resolve_object_xml(args)
    object_pose_npz = resolve_object_pose_npz(args, sequence_dir)
    pose_dir = resolve_pose_dir(args, sequence_dir)
    _, all_object_qpos = load_object_trajectory(
        sequence_dir,
        object_pose_npz,
        pose_dir,
    )
    physics_enabled = args.mode == "dynamic"
    add_support_plane = args.support_plane or physics_enabled
    model = build_combined_model(
        robot_xml,
        object_xml,
        all_object_qpos[0],
        add_support_plane,
        args.support_plane_z,
        physics_enabled,
        robot_override_yaml,
    )
    sequence = load_sequence(args, model, sequence_dir, object_pose_npz, pose_dir)

    if add_support_plane and args.support_plane_z is None:
        plane_offset = (
            DYNAMIC_SUPPORT_PLANE_OFFSET
            if physics_enabled
            else KINEMATIC_SUPPORT_PLANE_OFFSET
        )
        support_plane_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, SUPPORT_PLANE_GEOM)
        model.geom_pos[support_plane_id, 2] = infer_support_plane_z_from_object_xml(
            object_xml, sequence.object_qpos[0], plane_offset
        )

    return make_scene_info(model, sequence), sequence


def build_both_scene(args: argparse.Namespace) -> tuple[SceneInfo, SceneInfo, SequenceData]:
    sequence_dir = resolve_sequence_dir(args)
    robot_xml = resolve_robot_xml(args)
    robot_override_yaml = resolve_robot_override_yaml(args)
    object_xml = resolve_object_xml(args)
    object_pose_npz = resolve_object_pose_npz(args, sequence_dir)
    pose_dir = resolve_pose_dir(args, sequence_dir)
    _, all_object_qpos = load_object_trajectory(
        sequence_dir,
        object_pose_npz,
        pose_dir,
    )
    model = build_both_model(
        robot_xml,
        object_xml,
        all_object_qpos[0],
        args.support_plane_z,
        robot_override_yaml,
    )
    sequence = load_sequence(args, model, sequence_dir, object_pose_npz, pose_dir)

    if args.support_plane_z is None:
        support_plane_id = name_to_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"dyn_{SUPPORT_PLANE_GEOM}"
        )
        model.geom_pos[support_plane_id, 2] = infer_support_plane_z_from_object_xml(
            object_xml, sequence.object_qpos[0], DYNAMIC_SUPPORT_PLANE_OFFSET
        )

    kinematic_scene = make_scene_info(
        model, sequence, prefix="kin_", scene_offset=BOTH_KINEMATIC_OFFSET
    )
    dynamic_scene = make_scene_info(
        model, sequence, prefix="dyn_", scene_offset=BOTH_DYNAMIC_OFFSET
    )
    combined_qpos = dynamic_scene.initial_qpos.copy()
    write_robot_qpos(combined_qpos, kinematic_scene.joint_qposadr, sequence.robot_qpos[0])
    combined_qpos[
        kinematic_scene.object_qposadr : kinematic_scene.object_qposadr + 7
    ] = object_qpos_with_offset(sequence.object_qpos[0], kinematic_scene.scene_offset)
    combined_ctrl = dynamic_scene.initial_ctrl.copy()
    combined_ctrl[kinematic_scene.actuator_ids] = kinematic_scene.initial_ctrl[
        kinematic_scene.actuator_ids
    ]

    kinematic_scene = SceneInfo(
        model=model,
        object_body_id=kinematic_scene.object_body_id,
        object_qposadr=kinematic_scene.object_qposadr,
        joint_qposadr=kinematic_scene.joint_qposadr,
        actuator_ids=kinematic_scene.actuator_ids,
        scene_offset=kinematic_scene.scene_offset,
        initial_qpos=combined_qpos,
        initial_ctrl=combined_ctrl,
    )
    dynamic_scene = SceneInfo(
        model=model,
        object_body_id=dynamic_scene.object_body_id,
        object_qposadr=dynamic_scene.object_qposadr,
        joint_qposadr=dynamic_scene.joint_qposadr,
        actuator_ids=dynamic_scene.actuator_ids,
        scene_offset=dynamic_scene.scene_offset,
        initial_qpos=combined_qpos,
        initial_ctrl=combined_ctrl,
    )
    return kinematic_scene, dynamic_scene, sequence


def write_robot_qpos(
    qpos: np.ndarray, joint_qposadr: dict[str, int], robot_qpos: np.ndarray
) -> None:
    joint_names = ARM_ACTUATORS + HAND_ACTUATORS
    for actuator_index, joint_name in enumerate(joint_names):
        qpos[joint_qposadr[joint_name]] = robot_qpos[actuator_index]

    for joint_name, (parent_name, gain) in COUPLED_JOINTS.items():
        qpos[joint_qposadr[joint_name]] = (
            gain * qpos[joint_qposadr[parent_name]]
        )


def object_qpos_with_offset(object_qpos: np.ndarray, offset: np.ndarray) -> np.ndarray:
    shifted = object_qpos.copy()
    shifted[:3] += offset
    return shifted


def set_scene_ctrl(scene, data: mujoco.MjData, ctrl: np.ndarray) -> None:
    data.ctrl[scene.actuator_ids] = np.clip(
        ctrl,
        scene.model.actuator_ctrlrange[scene.actuator_ids, 0],
        scene.model.actuator_ctrlrange[scene.actuator_ids, 1],
    )


def set_robot_qpos(scene, data: mujoco.MjData, robot_qpos: np.ndarray) -> None:
    write_robot_qpos(data.qpos, scene.joint_qposadr, robot_qpos)


def set_kinematic_frame(scene, sequence, data: mujoco.MjData, index: int) -> None:
    data.qpos[:] = scene.initial_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = scene.initial_ctrl
    set_scene_ctrl(scene, data, sequence.controls[index])
    set_robot_qpos(scene, data, sequence.robot_qpos[index])
    data.qpos[scene.object_qposadr : scene.object_qposadr + 7] = object_qpos_with_offset(
        sequence.object_qpos[index], scene.scene_offset
    )
    mujoco.mj_forward(scene.model, data)


def print_debug_frame(
    scene,
    sequence,
    data: mujoco.MjData,
    index: int,
    prefix: str,
    tracked_object: bool,
) -> None:
    arm_ctrl = sequence.controls[index, :6]
    hand_ctrl = sequence.controls[index, 6:12]
    obj_qpos = sequence.object_qpos[index]
    sim_obj_pos = data.xpos[scene.object_body_id]
    obj_label = "tracked_obj_pos" if tracked_object else "sim_obj_pos"
    print(
        f"{prefix} frame={int(sequence.frame_ids[index])} "
        f"arm_ctrl(rad)={np.array2string(arm_ctrl, precision=3)} "
        f"hand_ctrl={np.array2string(hand_ctrl, precision=3)} "
        f"gt_obj_pos={np.array2string(obj_qpos[:3], precision=4)} "
        f"{obj_label}={np.array2string(sim_obj_pos, precision=4)}"
    )


def run_headless(scene, sequence, mode: str, print_every: int) -> None:
    data = mujoco.MjData(scene.model)
    for i in range(len(sequence.frame_ids)):
        set_kinematic_frame(scene, sequence, data, i)
        if i == 0 or i == len(sequence.frame_ids) - 1 or i % print_every == 0:
            print_debug_frame(scene, sequence, data, i, "kinematic", True)


def initialize_dynamic_data(scene, sequence) -> mujoco.MjData:
    data = mujoco.MjData(scene.model)
    data.qpos[:] = scene.initial_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = scene.initial_ctrl
    set_scene_ctrl(scene, data, sequence.controls[0])
    set_robot_qpos(scene, data, sequence.robot_qpos[0])
    data.qpos[scene.object_qposadr : scene.object_qposadr + 7] = object_qpos_with_offset(
        sequence.object_qpos[0], scene.scene_offset
    )
    mujoco.mj_forward(scene.model, data)
    return data


def step_dynamic_frame(
    scene,
    sequence,
    data: mujoco.MjData,
    index: int,
    previous_ctrl: np.ndarray,
) -> None:
    dt = sequence.dt
    sim_steps = max(1, int(round(dt / scene.model.opt.timestep)))
    current_ctrl = sequence.controls[index]
    ctrl_delta = current_ctrl - previous_ctrl

    for step in range(sim_steps):
        alpha = float(step + 1) / float(sim_steps)
        ctrl = previous_ctrl + alpha * ctrl_delta
        set_scene_ctrl(scene, data, ctrl)
        mujoco.mj_step(scene.model, data)


def run_dynamic_headless(scene, sequence, print_every: int) -> None:
    data = initialize_dynamic_data(scene, sequence)
    print_debug_frame(scene, sequence, data, 0, "dynamic", False)
    previous_ctrl = sequence.controls[0]
    for i in range(1, len(sequence.frame_ids)):
        step_dynamic_frame(scene, sequence, data, i, previous_ctrl)
        previous_ctrl = sequence.controls[i]
        if i == len(sequence.frame_ids) - 1 or i % print_every == 0:
            print_debug_frame(scene, sequence, data, i, "dynamic", False)


def set_kinematic_scene_frame(scene, sequence, data: mujoco.MjData, index: int) -> None:
    set_scene_ctrl(scene, data, sequence.controls[index])
    set_robot_qpos(scene, data, sequence.robot_qpos[index])
    data.qpos[scene.object_qposadr : scene.object_qposadr + 7] = object_qpos_with_offset(
        sequence.object_qpos[index], scene.scene_offset
    )


def reset_both_data_state(kinematic_scene, dynamic_scene, sequence, data: mujoco.MjData) -> None:
    data.qpos[:] = dynamic_scene.initial_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = dynamic_scene.initial_ctrl
    set_kinematic_scene_frame(kinematic_scene, sequence, data, 0)
    set_scene_ctrl(dynamic_scene, data, sequence.controls[0])
    set_robot_qpos(dynamic_scene, data, sequence.robot_qpos[0])
    data.qpos[
        dynamic_scene.object_qposadr : dynamic_scene.object_qposadr + 7
    ] = object_qpos_with_offset(sequence.object_qpos[0], dynamic_scene.scene_offset)
    mujoco.mj_forward(dynamic_scene.model, data)


def reset_both_data(kinematic_scene, dynamic_scene, sequence) -> mujoco.MjData:
    data = mujoco.MjData(dynamic_scene.model)
    reset_both_data_state(kinematic_scene, dynamic_scene, sequence, data)
    return data


def run_both_headless(
    kinematic_scene,
    dynamic_scene,
    sequence,
    print_every: int,
) -> None:
    data = reset_both_data(kinematic_scene, dynamic_scene, sequence)
    print_debug_frame(kinematic_scene, sequence, data, 0, "kinematic", True)
    print_debug_frame(dynamic_scene, sequence, data, 0, "dynamic", False)

    previous_ctrl = sequence.controls[0]
    for i in range(1, len(sequence.frame_ids)):
        step_dynamic_frame(dynamic_scene, sequence, data, i, previous_ctrl)
        set_kinematic_scene_frame(kinematic_scene, sequence, data, i)
        mujoco.mj_forward(dynamic_scene.model, data)
        previous_ctrl = sequence.controls[i]
        if i == len(sequence.frame_ids) - 1 or i % print_every == 0:
            print_debug_frame(kinematic_scene, sequence, data, i, "kinematic", True)
            print_debug_frame(dynamic_scene, sequence, data, i, "dynamic", False)


def make_restart_callback(restart_state: dict[str, bool]):
    def key_callback(keycode: int) -> None:
        if keycode == RESTART_KEY:
            restart_state["requested"] = True
            print("restart requested")

    return key_callback


def play_kinematic(scene, sequence, args: argparse.Namespace) -> None:
    data = mujoco.MjData(scene.model)
    wall_dt = sequence.dt / max(args.speed, 1e-6)
    restart_state = {"requested": False}

    with mujoco.viewer.launch_passive(
        scene.model,
        data,
        key_callback=make_restart_callback(restart_state),
    ) as viewer:
        while viewer.is_running():
            restart_state["requested"] = False
            for i in range(len(sequence.frame_ids)):
                if not viewer.is_running() or restart_state["requested"]:
                    break
                frame_start = time.time()
                set_kinematic_frame(scene, sequence, data, i)
                viewer.sync()
                if i == 0 or i == len(sequence.frame_ids) - 1 or i % args.print_every == 0:
                    print_debug_frame(scene, sequence, data, i, "kinematic", True)
                sleep_time = wall_dt - (time.time() - frame_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if restart_state["requested"]:
                continue
            if not args.loop:
                while viewer.is_running():
                    if restart_state["requested"]:
                        break
                    viewer.sync()
                    time.sleep(0.03)
                if restart_state["requested"]:
                    continue
                break


def play_dynamic(scene, sequence, args: argparse.Namespace) -> None:
    wall_dt = sequence.dt / max(args.speed, 1e-6)
    data = initialize_dynamic_data(scene, sequence)
    restart_state = {"requested": False}

    with mujoco.viewer.launch_passive(
        scene.model,
        data,
        key_callback=make_restart_callback(restart_state),
    ) as viewer:
        while viewer.is_running():
            restart_state["requested"] = False

            if args.visualize_contacts:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1
                # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
            data.qpos[:] = scene.initial_qpos
            data.qvel[:] = 0.0
            data.ctrl[:] = scene.initial_ctrl
            set_scene_ctrl(scene, data, sequence.controls[0])
            set_robot_qpos(scene, data, sequence.robot_qpos[0])
            data.qpos[scene.object_qposadr : scene.object_qposadr + 7] = object_qpos_with_offset(
                sequence.object_qpos[0], scene.scene_offset
            )
            mujoco.mj_forward(scene.model, data)
            viewer.sync()
            print_debug_frame(scene, sequence, data, 0, "dynamic", False)

            previous_ctrl = sequence.controls[0]
            for i in range(1, len(sequence.frame_ids)):
                if not viewer.is_running() or restart_state["requested"]:
                    break
                frame_start = time.time()
                step_dynamic_frame(scene, sequence, data, i, previous_ctrl)
                previous_ctrl = sequence.controls[i]
                viewer.sync()
                if i == len(sequence.frame_ids) - 1 or i % args.print_every == 0:
                    print_debug_frame(scene, sequence, data, i, "dynamic", False)
                sleep_time = wall_dt - (time.time() - frame_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if restart_state["requested"]:
                continue
            if not args.loop:
                while viewer.is_running():
                    if restart_state["requested"]:
                        break
                    viewer.sync()
                    time.sleep(0.03)
                if restart_state["requested"]:
                    continue
                break


def play_both(kinematic_scene, dynamic_scene, sequence, args: argparse.Namespace) -> None:
    wall_dt = sequence.dt / max(args.speed, 1e-6)
    data = reset_both_data(kinematic_scene, dynamic_scene, sequence)
    restart_state = {"requested": False}

    with mujoco.viewer.launch_passive(
        dynamic_scene.model,
        data,
        key_callback=make_restart_callback(restart_state),
    ) as viewer:
        while viewer.is_running():
            restart_state["requested"] = False

            if args.visualize_contacts:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

            reset_both_data_state(kinematic_scene, dynamic_scene, sequence, data)
            viewer.sync()
            print_debug_frame(kinematic_scene, sequence, data, 0, "kinematic", True)
            print_debug_frame(dynamic_scene, sequence, data, 0, "dynamic", False)

            previous_ctrl = sequence.controls[0]
            for i in range(1, len(sequence.frame_ids)):
                if not viewer.is_running() or restart_state["requested"]:
                    break
                frame_start = time.time()
                step_dynamic_frame(dynamic_scene, sequence, data, i, previous_ctrl)
                set_kinematic_scene_frame(kinematic_scene, sequence, data, i)
                mujoco.mj_forward(dynamic_scene.model, data)
                previous_ctrl = sequence.controls[i]
                viewer.sync()
                if i == len(sequence.frame_ids) - 1 or i % args.print_every == 0:
                    print_debug_frame(kinematic_scene, sequence, data, i, "kinematic", True)
                    print_debug_frame(dynamic_scene, sequence, data, i, "dynamic", False)
                sleep_time = wall_dt - (time.time() - frame_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if restart_state["requested"]:
                continue
            if not args.loop:
                while viewer.is_running():
                    if restart_state["requested"]:
                        break
                    viewer.sync()
                    time.sleep(0.03)
                if restart_state["requested"]:
                    continue
                break


def main() -> None:
    args = parse_args()
    if args.mode == "both":
        kinematic_scene, dynamic_scene, sequence = build_both_scene(args)
    else:
        scene, sequence = build_scene(args)
    sequence_dir = resolve_sequence_dir(args)
    robot_xml = resolve_robot_xml(args)
    robot_override_yaml = resolve_robot_override_yaml(args)
    object_xml = resolve_object_xml(args)
    pose_dir = resolve_pose_dir(args, sequence_dir)
    object_pose_npz = resolve_object_pose_npz(args, sequence_dir)
    object_source = pose_dir if pose_dir is not None else object_pose_npz

    print(
        f"object={args.object_name} episode={args.episode_number} "
        f"sequence_dir={sequence_dir} "
        f"robot_xml={robot_xml} "
        f"robot_override_yaml={robot_override_yaml} "
        f"object_xml={object_xml} "
        f"object_trajectory={object_source} "
        f"mode={args.mode} object_transform=inv(C2R) "
        f"frames={int(sequence.frame_ids[0])}..{int(sequence.frame_ids[-1])} "
        f"n={len(sequence.frame_ids)} dt={sequence.dt:.4f}s"
    )
    if args.mode == "both":
        print("showing kinematic and dynamic replay side by side in one MuJoCo window")
        print("dynamic robot is initialized from recorded qpos, then driven by MJCF PD actuators")
        print("support plane is forced on in the dynamic window")
    elif args.mode == "dynamic":
        print("object pose is initialized from inv(C2R) @ object_seq[0], then simulated with contacts")
        print("robot is initialized from recorded qpos, then driven by MJCF PD actuators")
        print("support plane is forced on in dynamic mode")
    else:
        print("robot is fixed in the MuJoCo world; object pose is inv(C2R) @ object_seq[t]")
    print(f"first object qpos={np.array2string(sequence.object_qpos[0], precision=5)}")
    print(f"first robot qpos={np.array2string(sequence.robot_qpos[0], precision=5)}")
    print(f"first robot ctrl={np.array2string(sequence.controls[0], precision=5)}")

    if args.headless:
        if args.mode == "both":
            run_both_headless(kinematic_scene, dynamic_scene, sequence, args.print_every)
        elif args.mode == "dynamic":
            run_dynamic_headless(scene, sequence, args.print_every)
        else:
            run_headless(scene, sequence, args.mode, args.print_every)
    elif args.mode == "both":
        play_both(kinematic_scene, dynamic_scene, sequence, args)
    elif args.mode == "dynamic":
        play_dynamic(scene, sequence, args)
    else:
        play_kinematic(scene, sequence, args)


if __name__ == "__main__":
    main()
