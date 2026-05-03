from __future__ import annotations

import torch

from .model import ResidualDynamicsModel


def main() -> None:
    torch.manual_seed(0)

    batch_size = 4
    hist_len = 8
    num_joints = 16
    num_points = 256
    sim_dim = 32

    q_t = torch.randn(batch_size, num_joints)
    qdot_t = torch.randn(batch_size, num_joints)
    u_t = torch.randn(batch_size, num_joints)
    obj_pose_t = torch.randn(batch_size, 7)
    obj_vel_t = torch.randn(batch_size, 3)
    obj_omega_t = torch.randn(batch_size, 3)
    sim_next_state = torch.randn(batch_size, sim_dim)
    q_hist = torch.randn(batch_size, hist_len, num_joints)
    u_hist = torch.randn(batch_size, hist_len, num_joints)
    object_points = torch.randn(batch_size, num_points, 3)

    model = ResidualDynamicsModel(num_joints=num_joints, sim_dim=sim_dim)
    outputs = model(
        q_t=q_t,
        qdot_t=qdot_t,
        u_t=u_t,
        obj_pose_t=obj_pose_t,
        obj_vel_t=obj_vel_t,
        obj_omega_t=obj_omega_t,
        sim_next_state=sim_next_state,
        q_hist=q_hist,
        u_hist=u_hist,
        object_points=object_points,
    )

    expected_shapes = {
        "delta_xi_obj": (batch_size, 6),
        "delta_v_obj": (batch_size, 3),
        "delta_omega_obj": (batch_size, 3),
        "slip_logit": (batch_size, 1),
        "contact_embedding": (batch_size, 128),
        "object_embedding": (batch_size, 128),
        "state_embedding": (batch_size, 128),
    }

    for key, expected_shape in expected_shapes.items():
        print(f"{key}: {tuple(outputs[key].shape)}")
        assert tuple(outputs[key].shape) == expected_shape


if __name__ == "__main__":
    main()
