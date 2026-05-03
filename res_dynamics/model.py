from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .utils import compute_tracking_residual_features


class MLP(nn.Module):
    """Small feed-forward network used by the prototype encoders and heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        *,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PointNetObjectEncoder(nn.Module):
    """PointNet-style object encoder for object point clouds.

    Input shape: [B, N, 3].
    Output shape: [B, object_dim].
    """

    def __init__(
        self,
        *,
        point_dim: int = 3,
        point_hidden_dims: Sequence[int] = (64, 128),
        point_feature_dim: int = 128,
        object_dim: int = 128,
        object_hidden_dims: Sequence[int] = (128,),
    ) -> None:
        super().__init__()
        self.object_dim = object_dim
        self.shared_point_mlp = MLP(point_dim, point_hidden_dims, point_feature_dim)
        self.object_mlp = MLP(point_feature_dim, object_hidden_dims, object_dim)

    def forward(self, object_points: Tensor) -> Tensor:
        if object_points.ndim != 3:
            raise ValueError(f"object_points must have shape [B, N, 3], got {object_points.shape}")

        point_features = self.shared_point_mlp(object_points)  # [B, N, point_feature_dim]
        pooled = point_features.max(dim=1).values  # [B, point_feature_dim]
        return self.object_mlp(pooled)  # [B, object_dim]


class StateActionSimEncoder(nn.Module):
    """Encodes current robot/object state, command, and nominal simulator prediction."""

    def __init__(
        self,
        *,
        num_joints: int,
        sim_dim: int,
        obj_pose_dim: int = 7,
        state_dim: int = 128,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.sim_dim = sim_dim
        self.obj_pose_dim = obj_pose_dim
        self.state_dim = state_dim

        input_dim = 3 * num_joints + obj_pose_dim + 3 + 3 + sim_dim
        self.encoder = MLP(input_dim, hidden_dims, state_dim)

    def forward(
        self,
        q_t: Tensor,
        qdot_t: Tensor,
        u_t: Tensor,
        obj_pose_t: Tensor,
        obj_vel_t: Tensor,
        obj_omega_t: Tensor,
        sim_next_state: Tensor,
    ) -> Tensor:
        # Each input is batched. Concatenated shape: [B, input_dim].
        x = torch.cat(
            (q_t, qdot_t, u_t, obj_pose_t, obj_vel_t, obj_omega_t, sim_next_state),
            dim=-1,
        )
        return self.encoder(x)  # [B, state_dim]


class ProprioceptiveContactEncoder(nn.Module):
    """Encodes contact evidence from command-tracking residual history.

    No raw tactile values are used. Contact evidence comes from features such as
    e_t = u_t - q_t and rho_t = abs(delta_q_t) / (abs(delta_u_t) + eps).
    """

    def __init__(
        self,
        *,
        num_joints: int,
        feature_dim: int = 6,
        joint_token_dim: int = 64,
        contact_dim: int = 128,
        joint_embed_dim: int = 0,
        joint_hidden_dims: Sequence[int] = (64,),
        gru_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.feature_dim = feature_dim
        self.joint_token_dim = joint_token_dim
        self.contact_dim = contact_dim
        self.joint_embed_dim = joint_embed_dim

        self.joint_embedding = (
            nn.Embedding(num_joints, joint_embed_dim) if joint_embed_dim > 0 else None
        )
        self.per_joint_mlp = MLP(
            feature_dim + joint_embed_dim,
            joint_hidden_dims,
            joint_token_dim,
        )
        self.temporal_encoder = nn.GRU(
            input_size=joint_token_dim,
            hidden_size=contact_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"features must have shape [B, K, J, F], got {features.shape}")

        batch_size, hist_len, num_joints, feature_dim = features.shape
        if num_joints != self.num_joints:
            raise ValueError(f"expected {self.num_joints} joints, got {num_joints}")
        if feature_dim != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}, got {feature_dim}")

        joint_inputs = features
        if self.joint_embedding is not None:
            joint_ids = torch.arange(num_joints, device=features.device)
            joint_emb = self.joint_embedding(joint_ids)  # [J, joint_embed_dim]
            joint_emb = joint_emb.view(1, 1, num_joints, self.joint_embed_dim)
            joint_emb = joint_emb.expand(batch_size, hist_len, -1, -1)
            joint_inputs = torch.cat((joint_inputs, joint_emb), dim=-1)

        flat = joint_inputs.reshape(batch_size * hist_len * num_joints, -1)
        tokens = self.per_joint_mlp(flat)
        tokens = tokens.view(batch_size, hist_len, num_joints, self.joint_token_dim)

        pooled_tokens = tokens.mean(dim=2)  # [B, K, joint_token_dim]
        _, hidden = self.temporal_encoder(pooled_tokens)
        return hidden[-1]  # [B, contact_dim]


class FusionResidualHead(nn.Module):
    """Predicts residual corrections from fused state, contact, and object embeddings."""

    def __init__(
        self,
        *,
        state_dim: int = 128,
        contact_dim: int = 128,
        object_dim: int = 128,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.head = MLP(state_dim + contact_dim + object_dim, hidden_dims, 13)

    def forward(
        self,
        state_embedding: Tensor,
        contact_embedding: Tensor,
        object_embedding: Tensor,
    ) -> dict[str, Tensor]:
        fused = torch.cat((state_embedding, contact_embedding, object_embedding), dim=-1)
        residual = self.head(fused)  # [B, 13]

        # delta_xi_obj is a 6D residual vector only. A later version should map it
        # through an SE(3) exponential and compose it with the simulator prediction.
        return {
            "delta_xi_obj": residual[:, 0:6],
            "delta_v_obj": residual[:, 6:9],
            "delta_omega_obj": residual[:, 9:12],
            "slip_logit": residual[:, 12:13],
        }


class ResidualDynamicsModel(nn.Module):
    """Residual world model for hand-object interaction dynamics."""

    def __init__(
        self,
        *,
        num_joints: int,
        sim_dim: int,
        obj_pose_dim: int = 7,
        state_dim: int = 128,
        contact_dim: int = 128,
        object_dim: int = 128,
        contact_feature_dim: int = 6,
        joint_token_dim: int = 64,
        joint_embed_dim: int = 0,
    ) -> None:
        super().__init__()
        self.state_encoder = StateActionSimEncoder(
            num_joints=num_joints,
            sim_dim=sim_dim,
            obj_pose_dim=obj_pose_dim,
            state_dim=state_dim,
        )
        self.contact_encoder = ProprioceptiveContactEncoder(
            num_joints=num_joints,
            feature_dim=contact_feature_dim,
            joint_token_dim=joint_token_dim,
            contact_dim=contact_dim,
            joint_embed_dim=joint_embed_dim,
        )
        self.object_encoder = PointNetObjectEncoder(object_dim=object_dim)
        self.residual_head = FusionResidualHead(
            state_dim=state_dim,
            contact_dim=contact_dim,
            object_dim=object_dim,
        )

    def forward(
        self,
        q_t: Tensor,
        qdot_t: Tensor,
        u_t: Tensor,
        obj_pose_t: Tensor,
        obj_vel_t: Tensor,
        obj_omega_t: Tensor,
        sim_next_state: Tensor,
        q_hist: Tensor,
        u_hist: Tensor,
        object_points: Tensor,
    ) -> dict[str, Tensor]:
        """Predict residual corrections to the nominal simulator transition.

        This model returns residual vectors. Applying them to form
        x_hat_real_{t+1} from x_sim_{t+1} is intentionally left to the rollout
        code; object pose composition should use an SE(3) update later.
        """
        state_embedding = self.state_encoder(
            q_t,
            qdot_t,
            u_t,
            obj_pose_t,
            obj_vel_t,
            obj_omega_t,
            sim_next_state,
        )
        contact_features = compute_tracking_residual_features(q_hist, u_hist)
        contact_embedding = self.contact_encoder(contact_features)
        object_embedding = self.object_encoder(object_points)

        outputs = self.residual_head(state_embedding, contact_embedding, object_embedding)
        outputs.update(
            {
                "contact_embedding": contact_embedding,
                "object_embedding": object_embedding,
                "state_embedding": state_embedding,
            }
        )
        return outputs
