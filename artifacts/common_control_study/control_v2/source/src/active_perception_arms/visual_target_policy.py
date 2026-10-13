"""RGB/proprio target estimation plus a prescribed, non-privileged insertion motion.

Unlike the end-to-end PPO benchmark, this structured baseline learns the target
pose with demonstration labels. Deployment reads RGB and measured robot state
only. No simulator object/goal state or variant id enters the actor.
"""

import torch

from .policy import VisualMemoryModel


class VisualTargetPolicy(VisualMemoryModel):
    def __init__(self, *args, episode_seconds=8.0, inspection_end=2.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_seconds = episode_seconds
        self.inspection_end = inspection_end
        self.start_tcp = None
        self.estimated_target = None
        self.register_buffer("action_low", torch.tensor([-0.07, -0.045, 0.055]))
        self.register_buffer("action_high", torch.tensor([0.07, 0.076, 0.09]))

    def predict_target(self, obs):
        return super().forward(obs)

    def action_from_target(self, prediction, obs):
        tcp = obs["proprio"][:, 36:39]
        t = obs["proprio"][:, -1] * self.episode_seconds
        fresh = t < 0.05
        if self.start_tcp is None or self.start_tcp.shape != tcp.shape:
            self.start_tcp = tcp.detach().clone()
            self.estimated_target = prediction.detach().clone()
        self.start_tcp[fresh] = tcp[fresh].detach()
        # Accumulate inspection estimates, then retain them while the hand occludes the socket.
        inspecting = (t < getattr(self, "inspection_end", 2.4))[:, None]
        self.estimated_target = torch.where(
            fresh[:, None], prediction.detach(), self.estimated_target
        )
        self.estimated_target = torch.where(
            inspecting,
            0.85 * self.estimated_target + 0.15 * prediction.detach(),
            self.estimated_target,
        )
        target = self.action_low + (self.estimated_target.clamp(-1, 1) + 1) * 0.5 * (
            self.action_high - self.action_low
        )
        if getattr(self, "fixed_insert_height", False):
            target[:, 2] = 0.061  # Fixed socket plane + calibrated TCP/grasp offset.
        target[:, :2] = torch.where((t < 1.6)[:, None], self.start_tcp[:, :2], target[:, :2])
        target[:, 2] = torch.where(t < 2.8, 0.09, torch.maximum(target[:, 2], tcp[:, 2] - 0.001))
        return (2 * (target - self.action_low) / (self.action_high - self.action_low) - 1).clamp(
            -1, 1
        )

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        return self.action_from_target(self.predict_target(obs), obs)

    def reset(self, dones=None, hidden_state=None):
        super().reset(dones, hidden_state)
        if dones is None:
            self.start_tcp = None
            self.estimated_target = None


class DenseImageEncoder(torch.nn.Module):
    """Retain local appearance as well as location for tiny prong/slot cues."""

    output_dim = 128
    output_channels = None

    def __init__(self, shape):
        super().__init__()
        h, w = shape
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, stride=2, padding=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.ELU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * ((h + 7) // 8) * ((w + 7) // 8), 128),
            torch.nn.ELU(),
        )

    def forward(self, images):
        return self.network(images)


class DenseVisualTargetPolicy(VisualTargetPolicy):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        encoders = torch.nn.ModuleDict(
            {
                name: DenseImageEncoder(obs[name].shape[-2:])
                for name in obs_groups[obs_set]
                if obs[name].ndim == 4
            }
        )
        super().__init__(obs, obs_groups, obs_set, output_dim, cnns=encoders, **kwargs)
        self.fixed_insert_height = True


class InspectionVisualTargetPolicy(DenseVisualTargetPolicy):
    """Estimate from pre-alignment images; no teacher-command shortcut."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inspection_end = 1.5

    def predict_target(self, obs):
        visual = dict(obs)
        visual["proprio"] = torch.zeros_like(obs["proprio"])
        return super().predict_target(visual)


class FactorizedInspectionPolicy(InspectionVisualTargetPolicy):
    """Learn socket XY and prong class separately from pre-alignment RGB."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp[-1] = torch.nn.Linear(self.mlp[-1].in_features, 6)
        self.register_buffer("fixture_center", torch.tensor([0.025, 0.055]))
        self.register_buffer(
            "variant_offsets", torch.tensor([[-0.015, 0], [0.015, 0], [0, -0.015], [0, 0.015]])
        )

    def predict_components(self, obs):
        visual = dict(obs)
        visual["proprio"] = torch.zeros_like(obs["proprio"])
        return self.mlp(self.get_latent(visual))

    def target_from_components(self, components):
        target = torch.zeros(len(components), 3, device=components.device, dtype=components.dtype)
        target[:, :2] = (
            self.fixture_center
            + 0.008 * components[:, :2]
            - self.variant_offsets[components[:, 2:].argmax(-1)]
        )
        target[:, 1] -= 0.0015
        target[:, 2] = 0.061
        return 2 * (target - self.action_low) / (self.action_high - self.action_low) - 1

    def predict_target(self, obs):
        return self.target_from_components(self.predict_components(obs))
