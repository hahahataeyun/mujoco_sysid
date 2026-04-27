"""Prototype friction identification for one Inspire F1 book grasp sequence.

This is intentionally a first working slice:
1. Load the recorded arm/hand controls and tracked object poses.
2. Build a MuJoCo scene by appending the book as a free body to the robot MJCF.
3. Replay the controls and compare simulated object poses to the tracked poses.
4. Optimize the book geom friction parameters.

The default path uses MuJoCo plus finite-difference gradients because it is fast
enough for this mesh-heavy prototype. An experimental MJX forward-mode path is
available with ``--backend mjx`` for later work on differentiable rollouts.

The synchronization assumptions are explicit CLI knobs because the current
sequence has no hand timestamp file and the object NPZ stores frame-indexed poses.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from scipy.spatial.transform import Rotation

from util.load_data import load_series, resample_to


ROOT = Path(__file__).resolve().parent
DEFAULT_SEQUENCE_DIR = Path(
    "/home/capture15/shared_data/capture/eccv2026/inspire_f1/book/0"
)
DEFAULT_ROBOT_XML = ROOT / "rsc/robot/xarm_inspire_f1.mjcf"
DEFAULT_OBJECT_URDF = ROOT / "rsc/object/book/book.urdf"
DEFAULT_SAVE_PATH = ROOT / "prototype_results/book0_friction_fit.npz"

OBJECT_BODY = "book"
OBJECT_GEOM = "book_geom"
OBJECT_FREEJOINT = "book_freejoint"
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
TIP_CONTACT_MESHES = {
    "thumb_force_sensor",
    "thumb_tip",
    "index_force_sensor",
    "index_tip",
    "middle_force_sensor",
    "middle_tip",
    "ring_force_sensor",
    "ring_tip",
    "little_force_sensor",
    "little_tip",
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
    mjx_model: mjx.Model | None
    object_body_id: int
    object_geom_id: int
    object_qposadr: int
    actuator_ids: np.ndarray
    joint_qposadr: dict[str, int]
    initial_qpos: np.ndarray
    initial_ctrl: np.ndarray
    initial_friction: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--object-urdf", type=Path, default=DEFAULT_OBJECT_URDF)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--backend", choices=("mujoco", "mjx"), default="mujoco")
    parser.add_argument("--fd-eps", type=float, default=1e-2)
    parser.add_argument("--hand-command-min", type=float, default=1000.0)
    parser.add_argument("--hand-command-max", type=float, default=2000.0)
    parser.add_argument("--rot-weight", type=float, default=0.02)
    parser.add_argument(
        "--contact-mode",
        choices=("tips", "hand", "all"),
        default="tips",
        help="Which robot geoms can contact the object in the generated scene.",
    )
    parser.add_argument("--friction-lower", type=float, nargs=3, default=(0.05, 0.0, 0.0))
    parser.add_argument("--friction-upper", type=float, nargs=3, default=(5.0, 0.2, 0.05))
    parser.add_argument("--initial-friction", type=float, nargs=3, default=None)
    parser.add_argument("--apply-c2r", dest="apply_c2r", action="store_true", default=True)
    parser.add_argument("--no-apply-c2r", dest="apply_c2r", action="store_false")
    parser.add_argument("--no-support-plane", action="store_true")
    parser.add_argument("--support-plane-z", type=float, default=None)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )
    return quat_wxyz / np.linalg.norm(quat_wxyz)


def load_object_trajectory(
    npz_path: Path, apply_c2r: bool, sequence_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load object poses as MuJoCo-world poses.

    The robot MJCF is kept fixed in the MuJoCo world.  The tracked object poses
    are recorded in the camera-world frame, so with ``apply_c2r=True`` each pose
    is transformed as:

        T_world_object = T_robot_camera @ T_camera_object

    where ``T_robot_camera`` is read from ``C2R.npy``.
    """
    npz = np.load(npz_path)
    keys = sorted(npz.files, key=lambda key: int(key.split("_")[-1]))
    frame_ids = np.array([int(key.split("_")[-1]) for key in keys], dtype=np.int64)
    transforms = np.stack([np.asarray(npz[key], dtype=np.float64) for key in keys])

    if apply_c2r:
        c2r_path = sequence_dir / "C2R.npy"
        if not c2r_path.exists():
            raise FileNotFoundError(f"--apply-c2r requested, but {c2r_path} is missing")
        c2r = np.load(c2r_path)
        transforms = np.einsum("ij,njk->nik", c2r, transforms)

    object_qpos = np.zeros((len(transforms), 7), dtype=np.float64)
    for i, transform in enumerate(transforms):
        object_qpos[i, :3] = transform[:3, 3]
        object_qpos[i, 3:] = matrix_to_wxyz(transform)
    return frame_ids, object_qpos


def load_sequence(args: argparse.Namespace, model: mujoco.MjModel) -> SequenceData:
    raw_dir = args.sequence_dir / "raw"
    # arm_qpos, arm_times = load_series(str(raw_dir / "arm"), ("action_qpos.npy",))
    arm_qpos, arm_times = load_series(str(raw_dir / "arm"), ("position.npy",))
    hand_commands, hand_times = load_series(str(raw_dir / "hand"), ("right_commands.npy",))
    hand_commands = np.asarray(hand_commands, dtype=np.float64)

    hand_time_file = raw_dir / "hand/time.npy"
    if not hand_time_file.exists():
        hand_times = np.linspace(arm_times[0], arm_times[-1], len(hand_commands))

    frame_ids, object_qpos = load_object_trajectory(
        args.sequence_dir / "object_tracking_result/obj_T_frames.npz",
        args.apply_c2r,
        args.sequence_dir,
    )
    object_times = np.linspace(arm_times[0], arm_times[-1], len(object_qpos))

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


def command_to_hand_ctrl(
    commands: np.ndarray, ctrlrange: np.ndarray, command_min: float, command_max: float
) -> np.ndarray:
    scale = (commands - command_min) / (command_max - command_min)
    scale = np.clip(scale, 0.0, 1.0)
    return ctrlrange[:, 0] + scale * (ctrlrange[:, 1] - ctrlrange[:, 0])


def read_object_defaults(object_urdf: Path) -> tuple[Path, np.ndarray]:
    root = ET.parse(object_urdf).getroot()
    mesh_node = root.find(".//collision/geometry/mesh")
    if mesh_node is None:
        mesh_node = root.find(".//visual/geometry/mesh")
    if mesh_node is None:
        raise ValueError(f"No mesh found in {object_urdf}")

    mesh_file = Path(mesh_node.attrib["filename"])
    if not mesh_file.is_absolute():
        mesh_file = (object_urdf.parent / mesh_file).resolve()

    friction = np.array([1.5, 0.01, 0.001], dtype=np.float64)
    mujoco_geom = root.find(".//mujoco/geom")
    if mujoco_geom is not None and "friction" in mujoco_geom.attrib:
        friction = np.fromstring(mujoco_geom.attrib["friction"], sep=" ", dtype=np.float64)
    return mesh_file, friction


def read_obj_vertices(mesh_path: Path) -> np.ndarray:
    vertices = []
    with mesh_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(item) for item in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices found in {mesh_path}")
    return np.asarray(vertices, dtype=np.float64)


def wxyz_to_rotation(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def infer_support_plane_z(object_mesh: Path, object_qpos: np.ndarray) -> float:
    vertices = read_obj_vertices(object_mesh)
    rotation = wxyz_to_rotation(object_qpos[3:])
    world_vertices = vertices @ rotation.T + object_qpos[:3]
    return float(np.min(world_vertices[:, 2]) - 0.002)


def keep_contact_geom(mesh_name: str | None, mode: str) -> bool:
    if mode == "all":
        return True
    if mesh_name is None:
        return False
    if mode == "tips":
        return mesh_name in TIP_CONTACT_MESHES
    if mode == "hand":
        return (
            mesh_name.startswith("right_")
            or mesh_name.endswith("_tip")
            or mesh_name.endswith("_force_sensor")
        )
    raise ValueError(f"Unknown contact mode: {mode}")


def set_robot_contact_filter(root: ET.Element, mode: str) -> None:
    if mode == "all":
        return
    for geom_node in root.find("worldbody").iter("geom"):
        mesh_name = geom_node.attrib.get("mesh")
        if keep_contact_geom(mesh_name, mode):
            geom_node.set("contype", "1")
            geom_node.set("conaffinity", "2")
        else:
            geom_node.set("contype", "0")
            geom_node.set("conaffinity", "0")


def build_combined_model(
    robot_xml: Path,
    object_urdf: Path,
    initial_object_qpos: np.ndarray,
    contact_mode: str,
    add_support_plane: bool,
    support_plane_z: float | None,
) -> tuple[mujoco.MjModel, np.ndarray]:
    object_mesh, initial_friction = read_object_defaults(object_urdf)
    root = ET.parse(robot_xml).getroot()

    for mesh_node in root.find("asset").findall("mesh"):
        mesh_file = mesh_node.attrib.get("file")
        if mesh_file and not Path(mesh_file).is_absolute():
            mesh_node.set("file", str((robot_xml.parent / mesh_file).resolve()))

    set_robot_contact_filter(root, contact_mode)

    ET.SubElement(root.find("asset"), "mesh", name="book_mesh", file=str(object_mesh))

    if add_support_plane:
        plane_z = (
            support_plane_z
            if support_plane_z is not None
            else infer_support_plane_z(object_mesh, initial_object_qpos)
        )
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name=SUPPORT_PLANE_GEOM,
            type="plane",
            pos=f"0 0 {plane_z}",
            size="0.8 0.8 0.02",
            friction="1.0 0.005 0.0001",
            contype="4" if contact_mode != "all" else "1",
            conaffinity="2" if contact_mode != "all" else "1",
            rgba="0.35 0.35 0.35 0.25",
        )

    object_body = ET.SubElement(
        root.find("worldbody"),
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
        mesh="book_mesh",
        mass="0.2",
        friction=" ".join(map(str, initial_friction)),
        contype="2" if contact_mode != "all" else "1",
        conaffinity="5" if contact_mode != "all" else "1",
        condim="3",
        solref="0.02 1",
        solimp="0.9 0.95 0.001",
        rgba="0.78 0.65 0.42 1",
    )

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, initial_friction


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
    object_frame_ids, object_qpos = load_object_trajectory(
        args.sequence_dir / "object_tracking_result/obj_T_frames.npz",
        args.apply_c2r,
        args.sequence_dir,
    )
    model, initial_friction = build_combined_model(
        args.robot_xml.resolve(),
        args.object_urdf.resolve(),
        object_qpos[0],
        args.contact_mode,
        not args.no_support_plane,
        args.support_plane_z,
    )
    sequence = load_sequence(args, model)

    if not args.no_support_plane and args.support_plane_z is None:
        support_plane_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, SUPPORT_PLANE_GEOM)
        object_mesh, _ = read_object_defaults(args.object_urdf.resolve())
        model.geom_pos[support_plane_id, 2] = infer_support_plane_z(
            object_mesh, sequence.object_qpos[0]
        )

    initial_qpos, initial_ctrl, joint_qposadr = make_initial_state(
        model, sequence.controls[0], sequence.object_qpos[0]
    )
    object_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    object_geom_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
    freejoint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_FREEJOINT)

    if args.initial_friction is not None:
        initial_friction = np.asarray(args.initial_friction, dtype=np.float64)
        model.geom_friction[object_geom_id] = initial_friction

    mjx_model = None
    if args.backend == "mjx":
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            mjx_model = mjx.put_model(model)

    return (
        SceneInfo(
            model=model,
            mjx_model=mjx_model,
            object_body_id=object_body_id,
            object_geom_id=object_geom_id,
            object_qposadr=model.jnt_qposadr[freejoint_id],
            actuator_ids=np.arange(model.nu),
            joint_qposadr=joint_qposadr,
            initial_qpos=initial_qpos,
            initial_ctrl=initial_ctrl,
            initial_friction=initial_friction,
        ),
        sequence,
    )


def friction_to_theta(
    friction: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    normalized = np.clip((friction - lower) / (upper - lower), 1e-4, 1.0 - 1e-4)
    return np.log(normalized) - np.log1p(-normalized)


def make_loss_fn(
    scene: SceneInfo,
    sequence: SequenceData,
    lower: np.ndarray,
    upper: np.ndarray,
    rot_weight: float,
):
    if scene.mjx_model is None:
        raise ValueError("MJX loss requested, but scene was built without an MJX model")
    mjx_model = scene.mjx_model
    object_geom_id = scene.object_geom_id
    object_body_id = scene.object_body_id
    sim_steps = max(1, int(round(sequence.dt / scene.model.opt.timestep)))

    qpos0 = jnp.asarray(scene.initial_qpos)
    qvel0 = jnp.zeros(scene.model.nv)
    ctrl0 = jnp.asarray(scene.initial_ctrl)
    controls = jnp.asarray(sequence.controls)
    gt_pos = jnp.asarray(sequence.object_pos)
    gt_quat = jnp.asarray(sequence.object_quat)
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)

    def theta_to_friction(theta: jax.Array) -> jax.Array:
        return lower + (upper - lower) * jax.nn.sigmoid(theta)

    def rollout_from_friction(friction: jax.Array) -> tuple[jax.Array, jax.Array]:
        model = mjx_model.replace(
            geom_friction=mjx_model.geom_friction.at[object_geom_id].set(friction)
        )
        data = mjx.make_data(model).replace(qpos=qpos0, qvel=qvel0, ctrl=ctrl0)
        data = mjx.forward(model, data)

        initial_pos = data.xpos[object_body_id]
        initial_quat = data.xquat[object_body_id]

        def one_frame(data, ctrl):
            data = data.replace(ctrl=ctrl)

            def one_step(step_data, _):
                return mjx.step(model, step_data), None

            data, _ = jax.lax.scan(one_step, data, xs=None, length=sim_steps)
            pos = data.xpos[object_body_id]
            quat = data.xquat[object_body_id]
            return data, (pos, quat)

        _, (pos_tail, quat_tail) = jax.lax.scan(one_frame, data, controls[1:])
        pos = jnp.concatenate([initial_pos[None], pos_tail], axis=0)
        quat = jnp.concatenate([initial_quat[None], quat_tail], axis=0)
        return pos, quat

    def loss_from_theta(theta: jax.Array) -> jax.Array:
        friction = theta_to_friction(theta)
        pos, quat = rollout_from_friction(friction)
        pos_loss = jnp.mean(jnp.sum((pos - gt_pos) ** 2, axis=-1))
        quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
        qdot = jnp.abs(jnp.sum(quat * gt_quat, axis=-1))
        rot_loss = jnp.mean((1.0 - jnp.clip(qdot, 0.0, 1.0)) ** 2)
        return pos_loss + rot_weight * rot_loss

    loss_fn = jax.jit(loss_from_theta)
    grad_fn = jax.jit(jax.jacfwd(loss_from_theta))

    def value_and_grad(theta: jax.Array) -> tuple[jax.Array, jax.Array]:
        return loss_fn(theta), grad_fn(theta)

    return (
        value_and_grad,
        jax.jit(lambda theta: theta_to_friction(theta)),
        jax.jit(lambda theta: rollout_from_friction(theta_to_friction(theta))),
        sim_steps,
    )


def theta_to_friction_np(theta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return lower + (upper - lower) / (1.0 + np.exp(-theta))


def pose_loss_np(
    sim_pos: np.ndarray,
    sim_quat: np.ndarray,
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    rot_weight: float,
) -> float:
    pos_loss = np.mean(np.sum((sim_pos - gt_pos) ** 2, axis=-1))
    sim_quat = sim_quat / np.linalg.norm(sim_quat, axis=-1, keepdims=True)
    qdot = np.abs(np.sum(sim_quat * gt_quat, axis=-1))
    rot_loss = np.mean((1.0 - np.clip(qdot, 0.0, 1.0)) ** 2)
    return float(pos_loss + rot_weight * rot_loss)


def rollout_mujoco(
    scene: SceneInfo, sequence: SequenceData, friction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model = scene.model
    model.geom_friction[scene.object_geom_id] = friction
    data = mujoco.MjData(model)
    data.qpos[:] = scene.initial_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = scene.initial_ctrl
    mujoco.mj_forward(model, data)

    sim_steps = max(1, int(round(sequence.dt / model.opt.timestep)))
    positions = [data.xpos[scene.object_body_id].copy()]
    quats = [data.xquat[scene.object_body_id].copy()]

    for ctrl in sequence.controls[1:]:
        data.ctrl[:] = ctrl
        for _ in range(sim_steps):
            mujoco.mj_step(model, data)
        positions.append(data.xpos[scene.object_body_id].copy())
        quats.append(data.xquat[scene.object_body_id].copy())

    return np.stack(positions), np.stack(quats)


def optimize_mujoco(scene: SceneInfo, sequence: SequenceData, args: argparse.Namespace):
    lower = np.asarray(args.friction_lower, dtype=np.float64)
    upper = np.asarray(args.friction_upper, dtype=np.float64)
    theta = friction_to_theta(scene.initial_friction, lower, upper)
    sim_steps = max(1, int(round(sequence.dt / scene.model.opt.timestep)))

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = []

    print(
        f"backend=mujoco frames={len(sequence.frame_ids)} frame_range={sequence.frame_ids[0]}.."
        f"{sequence.frame_ids[-1]} dt={sequence.dt:.4f}s sim_steps/frame={sim_steps}"
    )
    print(
        "initial friction",
        theta_to_friction_np(theta, lower, upper),
        "tracking first pos",
        sequence.object_pos[0],
    )

    def loss_at(theta_value: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        friction = theta_to_friction_np(theta_value, lower, upper)
        sim_pos, sim_quat = rollout_mujoco(scene, sequence, friction)
        loss = pose_loss_np(
            sim_pos, sim_quat, sequence.object_pos, sequence.object_quat, args.rot_weight
        )
        return loss, sim_pos, sim_quat

    sim_pos = sim_quat = None
    for i in range(args.iters + 1):
        loss, sim_pos, sim_quat = loss_at(theta)
        friction = theta_to_friction_np(theta, lower, upper)
        grad = np.zeros_like(theta)
        if i < args.iters:
            for j in range(len(theta)):
                delta = np.zeros_like(theta)
                delta[j] = args.fd_eps
                loss_plus, _, _ = loss_at(theta + delta)
                loss_minus, _, _ = loss_at(theta - delta)
                grad[j] = (loss_plus - loss_minus) / (2.0 * args.fd_eps)

        history.append((float(loss), friction.copy()))
        print(f"iter={i:03d} loss={loss:.8f} friction={friction} grad={grad}")

        if i == args.iters:
            break
        if not np.all(np.isfinite(grad)):
            print("stopping early: non-finite gradient")
            break

        step = i + 1
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        theta = theta - args.lr * m_hat / (np.sqrt(v_hat) + eps)

    assert sim_pos is not None and sim_quat is not None
    return theta_to_friction_np(theta, lower, upper), history[-1][0], sim_pos, sim_quat, history


def optimize_mjx(scene: SceneInfo, sequence: SequenceData, args: argparse.Namespace):
    lower = np.asarray(args.friction_lower, dtype=np.float64)
    upper = np.asarray(args.friction_upper, dtype=np.float64)
    theta = jnp.asarray(friction_to_theta(scene.initial_friction, lower, upper))

    value_and_grad, theta_to_friction, rollout_fn, sim_steps = make_loss_fn(
        scene, sequence, lower, upper, args.rot_weight
    )

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)

    print(
        f"backend=mjx frames={len(sequence.frame_ids)} frame_range={sequence.frame_ids[0]}.."
        f"{sequence.frame_ids[-1]} dt={sequence.dt:.4f}s sim_steps/frame={sim_steps}"
    )
    print(
        "initial friction",
        np.asarray(theta_to_friction(theta)),
        "tracking first pos",
        sequence.object_pos[0],
    )

    history = []
    for i in range(args.iters + 1):
        loss, grad = value_and_grad(theta)
        loss.block_until_ready()
        friction = theta_to_friction(theta)
        history.append((float(loss), np.asarray(friction)))
        print(
            f"iter={i:03d} loss={float(loss):.8f} "
            f"friction={np.asarray(friction)} grad={np.asarray(grad)}"
        )

        if i == args.iters:
            break
        if not np.all(np.isfinite(np.asarray(grad))):
            print("stopping early: non-finite gradient")
            break

        step = i + 1
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        theta = theta - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)

    sim_pos, sim_quat = rollout_fn(theta)
    sim_pos = np.asarray(sim_pos)
    sim_quat = np.asarray(sim_quat)
    final_friction = np.asarray(theta_to_friction(theta))
    final_loss = history[-1][0]
    return final_friction, final_loss, sim_pos, sim_quat, history


def optimize(scene: SceneInfo, sequence: SequenceData, args: argparse.Namespace):
    if args.backend == "mujoco":
        return optimize_mujoco(scene, sequence, args)
    return optimize_mjx(scene, sequence, args)


def save_results(
    path: Path,
    sequence: SequenceData,
    friction: np.ndarray,
    loss: float,
    sim_pos: np.ndarray,
    sim_quat: np.ndarray,
    history: list[tuple[float, np.ndarray]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        friction=friction,
        loss=loss,
        frame_ids=sequence.frame_ids,
        gt_pos=sequence.object_pos,
        gt_quat=sequence.object_quat,
        sim_pos=sim_pos,
        sim_quat=sim_quat,
        history_loss=np.array([item[0] for item in history]),
        history_friction=np.stack([item[1] for item in history]),
        apply_c2r=np.array(args.apply_c2r),
        start_frame=np.array(args.start_frame if args.start_frame is not None else -1),
        max_frames=np.array(args.max_frames if args.max_frames is not None else -1),
        stride=np.array(args.stride),
        contact_mode=np.array(args.contact_mode),
        support_plane_enabled=np.array(not args.no_support_plane),
        support_plane_z=np.array(
            args.support_plane_z if args.support_plane_z is not None else np.nan
        ),
    )


def main() -> None:
    args = parse_args()
    scene, sequence = build_scene(args)
    friction, loss, sim_pos, sim_quat, history = optimize(scene, sequence, args)

    final_pos_rmse = float(np.sqrt(np.mean(np.sum((sim_pos - sequence.object_pos) ** 2, axis=1))))
    print(f"final loss={loss:.8f} final_pos_rmse={final_pos_rmse:.6f} friction={friction}")

    if not args.no_save:
        save_results(args.save_path, sequence, friction, loss, sim_pos, sim_quat, history, args)
        print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()
