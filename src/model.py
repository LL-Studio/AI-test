from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


SELF_FEATURES = 15
PER_AGENT_FEATURES = 12
ENEMY_SLOTS = 2
ALLY_SLOTS = 2
AGENT_SLOTS = ENEMY_SLOTS + ALLY_SLOTS
RAY_COUNT = 8
OBSERVATION_DIM = SELF_FEATURES + RAY_COUNT + AGENT_SLOTS * PER_AGENT_FEATURES
ACTION_DIM = 6


@dataclass(frozen=True)
class PolicyAction:
    move_x: float
    move_y: float
    aim_x: float
    aim_y: float
    attack: bool
    parry: bool
    attack_score: float
    parry_score: float


class CombatPolicyNet(nn.Module):
    """Small deterministic combat policy used by evolutionary training.

    The network consumes a fixed-size ego-centric observation and emits:
      move vector, aim vector, attack logit, parry logit.
    """

    def __init__(self, obs_dim: int = OBSERVATION_DIM, hidden: int = 128) -> None:
        super().__init__()
        if obs_dim != OBSERVATION_DIM:
            raise ValueError(f"CombatPolicyNet expects obs_dim={OBSERVATION_DIM}, got {obs_dim}")

        self.self_enc = nn.Sequential(
            nn.Linear(SELF_FEATURES, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.ray_enc = nn.Sequential(
            nn.Linear(RAY_COUNT, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.agent_enc = nn.Sequential(
            nn.Linear(PER_AGENT_FEATURES, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Linear(48, 48),
            nn.GELU(),
        )

        fused_dim = 64 + 32 + 48 + 48
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, ACTION_DIM),
        )
        self._init_policy_biases()

    def _init_policy_biases(self) -> None:
        final = self.action_head[-1]
        if not isinstance(final, nn.Linear):
            return
        nn.init.normal_(final.weight, mean=0.0, std=0.02)
        nn.init.zeros_(final.bias)
        # Early random policies should produce fights instead of idle standoffs.
        final.bias.data[4] = 0.35
        final.bias.data[5] = -0.85

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x_self = observation[..., :SELF_FEATURES]
        ray_start = SELF_FEATURES
        ray_end = ray_start + RAY_COUNT
        x_ray = observation[..., ray_start:ray_end]
        x_agents = observation[..., ray_end:].reshape(
            *observation.shape[:-1],
            AGENT_SLOTS,
            PER_AGENT_FEATURES,
        )

        enemy_slots = x_agents[..., :ENEMY_SLOTS, :]
        ally_slots = x_agents[..., ENEMY_SLOTS:, :]

        self_h = self.self_enc(x_self)
        ray_h = self.ray_enc(x_ray)
        enemy_h = self._masked_slot_mean(self.agent_enc(enemy_slots), enemy_slots[..., 0])
        ally_h = self._masked_slot_mean(self.agent_enc(ally_slots), ally_slots[..., 0])
        fused = torch.cat([self_h, ray_h, enemy_h, ally_h], dim=-1)
        return self.action_head(self.fuse(fused))

    @staticmethod
    def _masked_slot_mean(encoded: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        mask = present.unsqueeze(-1).clamp(0.0, 1.0)
        total = (encoded * mask).sum(dim=-2)
        count = mask.sum(dim=-2).clamp(min=1.0)
        return total / count

    @torch.no_grad()
    def act(
        self,
        observation: list[float] | torch.Tensor,
        *,
        attack_threshold: float = 0.52,
        parry_threshold: float = 0.68,
    ) -> PolicyAction:
        if not isinstance(observation, torch.Tensor):
            obs = torch.tensor(observation, dtype=torch.float32)
        else:
            obs = observation.to(dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        out = self(obs).squeeze(0)
        move = torch.tanh(out[0:2])
        aim = torch.tanh(out[2:4])
        attack_score = torch.sigmoid(out[4]).item()
        parry_score = torch.sigmoid(out[5]).item()
        return PolicyAction(
            move_x=float(move[0].item()),
            move_y=float(move[1].item()),
            aim_x=float(aim[0].item()),
            aim_y=float(aim[1].item()),
            attack=attack_score >= attack_threshold,
            parry=parry_score >= parry_threshold,
            attack_score=float(attack_score),
            parry_score=float(parry_score),
        )


def build_model() -> CombatPolicyNet:
    return CombatPolicyNet()
