from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pd_id.data import WindowBatch
from pd_id.model import InitialParameters, RobotLayout


@dataclass(frozen=True)
class LossWeights:
    position: float = 100.0
    velocity: float = 10.0
    torque: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    sim_substeps: int
    learning_rate: float
    grad_mode: str = "reverse"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip_norm: float = 100.0
    positive_floor: float = 1e-8


def require_jax_mjx():
    try:
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "JAX, MuJoCo, and MJX are required for optimization. Install `jax`, "
            "`jaxlib`, and `mujoco` in this environment."
        ) from exc
    return jax, jnp, mujoco, mjx


def _inverse_softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 20.0, x, np.log(np.expm1(np.maximum(x, 1e-12))))


def init_raw_params(initial: InitialParameters, floor: float) -> dict[str, np.ndarray]:
    return {
        "actuator_kp": _inverse_softplus_np(initial.actuator_kp - floor),
        "actuator_dampratio": _inverse_softplus_np(initial.actuator_dampratio - floor),
        "joint_frictionloss": _inverse_softplus_np(initial.joint_frictionloss - floor),
        "joint_damping": _inverse_softplus_np(initial.joint_damping - floor),
        "joint_armature": _inverse_softplus_np(initial.joint_armature - floor),
    }


def materialize_params(raw_params: dict[str, Any], floor: float) -> dict[str, Any]:
    jax, _, _, _ = require_jax_mjx()
    return {key: jax.nn.softplus(value) + floor for key, value in raw_params.items()}


def materialize_params_np(raw_params: dict[str, Any], floor: float) -> dict[str, np.ndarray]:
    params = materialize_params(raw_params, floor)
    return {key: np.asarray(value, dtype=np.float64) for key, value in params.items()}


def _replace_model(model, **kwargs):
    if hasattr(model, "replace"):
        return model.replace(**kwargs)
    return model.tree_replace(kwargs)


def build_trainer(
    mujoco_model,
    layout: RobotLayout,
    initial: InitialParameters,
    weights: LossWeights,
    config: TrainConfig,
):
    jax, jnp, _, mjx = require_jax_mjx()

    base_model = mjx.put_model(mujoco_model)
    base_data = mjx.make_data(base_model)

    dof_ids = jnp.asarray(layout.dof_ids)
    qpos_ids = jnp.asarray(layout.qpos_ids)
    actuator_ids = jnp.asarray(layout.actuator_ids)
    velocity_gain_scale = jnp.asarray(initial.velocity_gain_scale)

    def apply_params(raw_params):
        params = materialize_params(raw_params, config.positive_floor)
        kp = params["actuator_kp"]
        dampratio = params["actuator_dampratio"]
        velocity_gain = dampratio * velocity_gain_scale * jnp.sqrt(kp)

        actuator_gainprm = base_model.actuator_gainprm.at[actuator_ids, 0].set(kp)
        actuator_biasprm = base_model.actuator_biasprm.at[actuator_ids, 1].set(-kp)
        actuator_biasprm = actuator_biasprm.at[actuator_ids, 2].set(-velocity_gain)
        dof_frictionloss = base_model.dof_frictionloss.at[dof_ids].set(
            params["joint_frictionloss"]
        )
        dof_damping = base_model.dof_damping.at[dof_ids].set(params["joint_damping"])
        dof_armature = base_model.dof_armature.at[dof_ids].set(params["joint_armature"])

        return _replace_model(
            base_model,
            actuator_gainprm=actuator_gainprm,
            actuator_biasprm=actuator_biasprm,
            dof_frictionloss=dof_frictionloss,
            dof_damping=dof_damping,
            dof_armature=dof_armature,
        )

    def rollout_one(model, qpos0, qvel0, controls):
        qpos = base_data.qpos.at[qpos_ids].set(qpos0)
        qvel = base_data.qvel.at[dof_ids].set(qvel0)
        data = _replace_model(base_data, qpos=qpos, qvel=qvel, ctrl=controls[0])
        data = mjx.forward(model, data)

        def integrate_sample(data, ctrl):
            data = _replace_model(data, ctrl=ctrl)

            def substep(_, inner_data):
                return mjx.step(model, inner_data)

            data = jax.lax.fori_loop(0, config.sim_substeps, substep, data)
            tau = data.qfrc_actuator[dof_ids]
            return data, (data.qpos[qpos_ids], data.qvel[dof_ids], tau)

        _, outputs = jax.lax.scan(integrate_sample, data, controls)
        return outputs

    batched_rollout = jax.vmap(rollout_one, in_axes=(None, 0, 0, 0))

    def loss_fn(raw_params, batch: dict[str, Any]):
        model = apply_params(raw_params)
        pred_qpos, pred_qvel, pred_tau = batched_rollout(
            model, batch["qpos0"], batch["qvel0"], batch["controls"]
        )
        mask = batch["mask"][..., None]
        denom = jnp.maximum(jnp.sum(mask) * pred_qpos.shape[-1], 1.0)
        joint_denom = jnp.maximum(jnp.sum(mask), 1.0)

        pos_loss = jnp.sum(mask * jnp.square(pred_qpos - batch["target_qpos"])) / denom
        vel_loss = jnp.sum(mask * jnp.square(pred_qvel - batch["target_qvel"])) / denom
        tau_loss = jnp.sum(mask * jnp.square(pred_tau - batch["target_tau"])) / denom
        pos_loss_joint = jnp.sum(
            mask * jnp.square(pred_qpos - batch["target_qpos"]), axis=(0, 1)
        ) / joint_denom
        vel_loss_joint = jnp.sum(
            mask * jnp.square(pred_qvel - batch["target_qvel"]), axis=(0, 1)
        ) / joint_denom
        tau_loss_joint = jnp.sum(
            mask * jnp.square(pred_tau - batch["target_tau"]), axis=(0, 1)
        ) / joint_denom
        total_loss_joint = (
            weights.position * pos_loss_joint
            + weights.velocity * vel_loss_joint
            + weights.torque * tau_loss_joint
        )
        total = weights.position * pos_loss + weights.velocity * vel_loss + weights.torque * tau_loss
        metrics = {
            "loss": total,
            "pos_loss": pos_loss,
            "vel_loss": vel_loss,
            "tau_loss": tau_loss,
            "joint/total_loss": total_loss_joint,
            "joint/pos_loss": pos_loss_joint,
            "joint/vel_loss": vel_loss_joint,
            "joint/tau_loss": tau_loss_joint,
        }
        return total, metrics

    def tree_zeros_like(tree):
        return jax.tree_util.tree_map(jnp.zeros_like, tree)

    def global_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))

    def adam_init(raw_params):
        params = jax.tree_util.tree_map(jnp.asarray, raw_params)
        return {
            "params": params,
            "m": tree_zeros_like(params),
            "v": tree_zeros_like(params),
            "step": jnp.asarray(0, dtype=jnp.int32),
        }

    @jax.jit
    def train_step(state, batch):
        if config.grad_mode == "forward":
            loss, metrics = loss_fn(state["params"], batch)
            grads = jax.jacfwd(lambda params: loss_fn(params, batch)[0])(state["params"])
        else:
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state["params"], batch
            )
        norm = global_norm(grads)
        if config.grad_clip_norm > 0.0:
            scale = jnp.minimum(1.0, config.grad_clip_norm / (norm + 1e-12))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)

        step = state["step"] + 1
        m = jax.tree_util.tree_map(
            lambda m, g: config.beta1 * m + (1.0 - config.beta1) * g, state["m"], grads
        )
        v = jax.tree_util.tree_map(
            lambda v, g: config.beta2 * v + (1.0 - config.beta2) * jnp.square(g),
            state["v"],
            grads,
        )
        lr_t = (
            config.learning_rate
            * jnp.sqrt(1.0 - config.beta2**step)
            / (1.0 - config.beta1**step)
        )
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - lr_t * m / (jnp.sqrt(v) + config.eps),
            state["params"],
            m,
            v,
        )
        metrics = dict(metrics)
        metrics["grad_norm"] = norm
        return {"params": params, "m": m, "v": v, "step": step}, metrics

    @jax.jit
    def evaluate(raw_params, batch):
        _, metrics = loss_fn(raw_params, batch)
        return metrics

    @jax.jit
    def rollout(raw_params, batch):
        model = apply_params(raw_params)
        pred_qpos, pred_qvel, pred_tau = batched_rollout(
            model, batch["qpos0"], batch["qvel0"], batch["controls"]
        )
        return pred_qpos, pred_qvel, pred_tau

    return adam_init, train_step, evaluate, rollout


def batch_to_jax(batch: WindowBatch) -> dict[str, Any]:
    _, jnp, _, _ = require_jax_mjx()
    return {
        "qpos0": jnp.asarray(batch.qpos0),
        "qvel0": jnp.asarray(batch.qvel0),
        "controls": jnp.asarray(batch.controls),
        "target_qpos": jnp.asarray(batch.target_qpos),
        "target_qvel": jnp.asarray(batch.target_qvel),
        "target_tau": jnp.asarray(batch.target_tau),
        "mask": jnp.asarray(batch.mask),
    }
