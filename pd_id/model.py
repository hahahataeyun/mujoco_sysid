from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_ROOTS = (
    REPO_ROOT / "pd_id/initial_mjcf",
    REPO_ROOT / "rsc/robot",
    REPO_ROOT / "rsc/curobo/content/assets/robot/inspire_description",
)


@dataclass(frozen=True)
class RobotLayout:
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    actuator_ids: np.ndarray


@dataclass(frozen=True)
class InitialParameters:
    actuator_kp: np.ndarray
    actuator_dampratio: np.ndarray
    joint_frictionloss: np.ndarray
    joint_damping: np.ndarray
    joint_armature: np.ndarray
    velocity_gain_scale: np.ndarray


def require_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo is required for PD identification. Install the `mujoco` Python "
            "package in this environment."
        ) from exc
    return mujoco


def _resolve_asset_file(file_name: str, xml_dir: Path, asset_roots: tuple[Path, ...]) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path

    candidates = [xml_dir / path]
    candidates.extend(root / path for root in asset_roots)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def _xml_with_resolved_assets(xml_path: Path, asset_roots: tuple[Path, ...]) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    changed = False
    xml_dir = xml_path.parent

    for elem in root.findall(".//*[@file]"):
        original = elem.get("file")
        if not original:
            continue
        resolved = _resolve_asset_file(original, xml_dir, asset_roots)
        if resolved != Path(original):
            elem.set("file", str(resolved))
            changed = True

    if not changed:
        return str(xml_path)

    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".mjcf", prefix=f"{xml_path.stem}_resolved_", delete=False
    )
    with handle:
        tree.write(handle, encoding="unicode", xml_declaration=False)
    return handle.name


def load_mujoco_model(
    xml_path: Path, *, resolve_assets: bool = True, asset_roots: tuple[Path, ...] = DEFAULT_MESH_ROOTS
):
    mujoco = require_mujoco()
    xml_to_load = (
        _xml_with_resolved_assets(xml_path, asset_roots) if resolve_assets else str(xml_path)
    )
    return mujoco.MjModel.from_xml_path(xml_to_load)


def arm_layout(
    model, joint_names: tuple[str, ...] = ARM_JOINTS, *, require_actuators: bool = True
) -> RobotLayout:
    mujoco = require_mujoco()
    qpos_ids = []
    dof_ids = []
    actuator_ids = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint not found in model: {name}")
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0 and require_actuators:
            raise ValueError(f"actuator not found in model: {name}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        dof_ids.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(actuator_id)

    return RobotLayout(
        joint_names=joint_names,
        actuator_names=joint_names,
        qpos_ids=np.asarray(qpos_ids, dtype=np.int32),
        dof_ids=np.asarray(dof_ids, dtype=np.int32),
        actuator_ids=np.asarray(actuator_ids, dtype=np.int32),
    )


def initial_parameters(model, layout: RobotLayout) -> InitialParameters:
    actuator_ids = layout.actuator_ids
    dof_ids = layout.dof_ids

    kp = np.asarray(model.actuator_gainprm[actuator_ids, 0], dtype=np.float64)
    kp = np.maximum(kp, 1e-6)

    velocity_gain = -np.asarray(model.actuator_biasprm[actuator_ids, 2], dtype=np.float64)
    velocity_gain = np.maximum(velocity_gain, 0.0)
    dampratio = np.ones_like(kp)

    armature = np.asarray(model.dof_armature[dof_ids], dtype=np.float64)

    fallback_scale = 2.0 * np.sqrt(np.maximum(armature, 1e-6))
    scale = np.where(velocity_gain > 1e-9, velocity_gain / np.sqrt(kp), fallback_scale)
    scale = np.maximum(scale, 1e-6)

    return InitialParameters(
        actuator_kp=kp,
        actuator_dampratio=dampratio,
        joint_frictionloss=np.maximum(np.asarray(model.dof_frictionloss[dof_ids]), 1e-8),
        joint_damping=np.maximum(np.asarray(model.dof_damping[dof_ids]), 1e-8),
        joint_armature=np.maximum(armature, 1e-8),
        velocity_gain_scale=scale,
    )


def write_fitted_xml(
    source_xml: Path,
    output_xml: Path,
    *,
    params: dict[str, np.ndarray],
    joint_names: tuple[str, ...] = ARM_JOINTS,
    timestep: float | None = None,
    resolve_assets: bool = True,
    asset_roots: tuple[Path, ...] = DEFAULT_MESH_ROOTS,
) -> None:
    tree = ET.parse(source_xml)
    root = tree.getroot()

    if timestep is not None:
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("timestep", f"{float(timestep):.12g}")

    if resolve_assets:
        xml_dir = source_xml.parent
        for elem in root.findall(".//*[@file]"):
            original = elem.get("file")
            if original:
                elem.set("file", str(_resolve_asset_file(original, xml_dir, asset_roots)))

    joints = {elem.attrib.get("name"): elem for elem in root.iter("joint")}
    actuators = {elem.attrib.get("name"): elem for elem in root.iter("position")}

    for i, name in enumerate(joint_names):
        joint = joints.get(name)
        if joint is None:
            raise ValueError(f"joint not found while writing XML: {name}")
        joint.set("frictionloss", f"{float(params['joint_frictionloss'][i]):.9g}")
        joint.set("damping", f"{float(params['joint_damping'][i]):.9g}")
        joint.set("armature", f"{float(params['joint_armature'][i]):.9g}")

        actuator = actuators.get(name)
        if actuator is None:
            raise ValueError(f"actuator not found while writing XML: {name}")
        actuator.set("kp", f"{float(params['actuator_kp'][i]):.9g}")
        actuator.set("dampratio", f"{float(params['actuator_dampratio'][i]):.9g}")

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)
