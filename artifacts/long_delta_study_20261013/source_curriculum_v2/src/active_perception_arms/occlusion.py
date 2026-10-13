"""World-space optical sweeps, independent of the camera policy and hidden state.

Native and Warp use the same distributions and trajectory equation. Their RNG
implementations differ; matching a sampled parameter set gives matching poses.
This intervention removes images without changing object state or contacts.
"""

from dataclasses import replace

import numpy as np
import torch


def with_occlusion(cfg, mode, **changes):
    """Switch a saved experiment's optical intervention, clearing inactive knobs.

    Suitable for actor-only evaluation controls. Training resume must still use
    its unmodified saved experiment and scientific interaction budget.
    """
    mode = mode or cfg.occlusion
    if mode != "dynamic":
        for name in (
            "occlusion_revision",
            "dynamic_onset_range",
            "dynamic_duration_range",
            "dynamic_center",
            "dynamic_center_jitter",
            "dynamic_travel_range",
            "dynamic_panel_half_size",
            "dynamic_panel_yaw",
        ):
            changes.setdefault(name, None)
    return replace(cfg, occlusion=mode, **changes)


def sample_dynamic_numpy(rng, cfg):
    return {
        "onset": float(rng.uniform(*cfg.dynamic_onset_range)),
        "duration": float(rng.uniform(*cfg.dynamic_duration_range)),
        "side": float(rng.choice([-1, 1])),
        "center": np.asarray(cfg.dynamic_center)
        + rng.uniform(-1, 1, 3) * np.asarray(cfg.dynamic_center_jitter),
        "travel": float(rng.uniform(*cfg.dynamic_travel_range)),
    }


def sample_dynamic_torch(n, cfg, device):
    def uniform(bounds):
        low, high = bounds
        return low + torch.rand(n, device=device) * (high - low)

    return {
        "onset": uniform(cfg.dynamic_onset_range),
        "duration": uniform(cfg.dynamic_duration_range),
        "side": torch.randint(0, 2, (n,), device=device).float() * 2 - 1,
        "center": torch.tensor(cfg.dynamic_center, device=device)
        + (torch.rand(n, 3, device=device) * 2 - 1)
        * torch.tensor(cfg.dynamic_center_jitter, device=device),
        "travel": uniform(cfg.dynamic_travel_range),
    }


def dynamic_position_numpy(time, onset, duration, side, center, travel):
    u = np.clip((time - onset) / duration, 0, 1)
    position = np.asarray(center, dtype=float).copy()
    position[0] += side * travel * (u - 0.5)
    if not onset <= time <= onset + duration:
        position[2] = -1.0
    return position


def dynamic_position_torch(time, onset, duration, side, center, travel):
    # No host transfers or data-dependent branching: usable during CUDA capture.
    u = ((time - onset) / duration).clamp(0, 1)
    position = center.clone()
    position[:, 0] += side * travel * (u - 0.5)
    visible = (time >= onset) & (time <= onset + duration)
    position[:, 2] = torch.where(visible, center[:, 2], -1.0)
    return position
