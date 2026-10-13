"""Shared recurrent visual actor and compact PPO storage for RSL-RL 5.2."""

from copy import deepcopy

import torch
import torch.nn as nn
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
from rsl_rl.algorithms import PPO
from rsl_rl.modules import RNN
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, unpad_trajectories

# Auxiliary perception targets (plug): hidden variant one-hot, then scaled
# goal-minus-TCP and socket offsets. Labels come from the privileged "aux"
# observation group, which is stored for training but never an actor input.
AUX_CLASSES = 4


def aux_head(latent_dim, aux_dim):
    return nn.Sequential(nn.Linear(latent_dim, 128), nn.ELU(), nn.Linear(128, aux_dim))


class VisualMemoryModel(SpatialSoftmaxCNNModel):
    """Encode views and proprioception once; GRU state is shared by both arm outputs."""

    is_recurrent = True

    def __init__(
        self, *args, rnn_type="gru", rnn_hidden_dim=128, rnn_num_layers=1, aux_dim=0, **kwargs
    ):
        self.latent_dim = rnn_hidden_dim
        super().__init__(*args, **kwargs)
        self.rnn = RNN(super()._get_latent_dim(), rnn_hidden_dim, rnn_num_layers, rnn_type)
        # Optional auxiliary perception head on the recurrent latent (training only).
        self.aux_head = aux_head(self.latent_dim, aux_dim) if aux_dim else None
        self.last_latent = None

    def _get_latent_dim(self):
        return self.latent_dim

    def encode(self, obs, masks=None):
        one_d = self.obs_normalizer(torch.cat([obs[g] for g in self.obs_groups], dim=-1))
        result = [one_d]
        for group in self.obs_groups_2d:
            images = obs[group]
            leading = images.shape[:-3]
            flat = images.reshape(-1, *images.shape[-3:])
            if masks is None:
                encoded = self.cnns[group](flat.float() / 255.0)
            else:
                # Avoid convolving zero padding in fragmented recurrent rollouts.
                valid = masks.reshape(-1)
                values = self.cnns[group](flat[valid].float() / 255.0)
                encoded = values.new_zeros(flat.shape[0], self.cnns[group].output_dim)
                encoded[valid] = values
            result.append(encoded.reshape(*leading, -1))
        return torch.cat(result, dim=-1)

    def get_latent(self, obs, masks=None, hidden_state=None):
        latent = self.rnn(self.encode(obs, masks), masks, hidden_state).squeeze(0)
        self.last_latent = latent
        return latent

    def reset(self, dones=None, hidden_state=None):
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self):
        return self.rnn.hidden_state

    def detach_hidden_state(self, dones=None):
        self.rnn.detach_hidden_state(dones)

    def as_jit(self):
        raise NotImplementedError(
            "Use the recurrent checkpoint evaluator; a stateless export would lose memory"
        )

    def as_onnx(self, verbose=False):
        raise NotImplementedError("Use the recurrent checkpoint evaluator")


class CompactStorage(RolloutStorage):
    """Allocate image rollouts as uint8 without first allocating float images."""

    def __init__(self, num_envs, steps, obs, actions_shape, device):
        images = [key for key, value in obs.items() if value.dtype == torch.uint8]
        super().__init__("rl", num_envs, steps, obs.exclude(*images), actions_shape, device)
        for key in images:
            self.observations[key] = torch.zeros(
                steps, *obs[key].shape, dtype=torch.uint8, device=device
            )


class VisualFeedForwardModel(SpatialSoftmaxCNNModel):
    """Same image encoder with no actor memory, for a separately trained ablation."""

    encode = VisualMemoryModel.encode

    def __init__(
        self, *args, rnn_type=None, rnn_hidden_dim=None, rnn_num_layers=None, aux_dim=0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.aux_head = aux_head(self._get_latent_dim(), aux_dim) if aux_dim else None
        self.last_latent = None

    def get_latent(self, obs, masks=None, hidden_state=None):
        # MLPModel.forward has already removed recurrent padding for this model.
        latent = self.encode(obs)
        self.last_latent = latent
        return latent


class CompactPPO(PPO):
    """Compact image storage and bootstrap evaluation without consuming RNN state.

    ``aux_coef`` > 0 adds a supervised auxiliary perception loss on the actor's
    latent (privileged labels from the stored "aux" observation group). The
    update below is RSL-RL 5.2's PPO.update with that single extra term; with
    ``aux_coef`` = 0 it performs the upstream computation.
    """

    def __init__(self, *args, aux_coef=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_coef = float(aux_coef)
        if self.aux_coef and getattr(self.actor, "aux_head", None) is None:
            raise ValueError("aux_coef requires an actor built with aux_dim > 0")

    def aux_loss(self, batch):
        latent = self.actor.last_latent
        labels = batch.observations["aux"]
        if batch.masks is not None:
            labels = unpad_trajectories(labels, batch.masks)
        prediction = self.actor.aux_head(latent).reshape(-1, labels.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])
        classes = labels[:, :AUX_CLASSES].argmax(-1)
        logits = prediction[:, :AUX_CLASSES]
        loss = torch.nn.functional.cross_entropy(logits, classes)
        loss = loss + torch.nn.functional.mse_loss(
            prediction[:, AUX_CLASSES:], labels[:, AUX_CLASSES:]
        )
        accuracy = (logits.argmax(-1) == classes).float().mean()
        return loss, accuracy

    def update(self):
        if not self.aux_coef:
            return super().update()
        if self.rnd or self.symmetry or self.is_multi_gpu:
            raise NotImplementedError("Auxiliary loss is implemented for single-GPU PPO only")
        mean_value_loss = mean_surrogate_loss = mean_entropy = 0.0
        mean_aux_loss = mean_aux_accuracy = 0.0
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
            )
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
            aux_loss, aux_accuracy = self.aux_loss(batch)
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.aux_coef * aux_loss
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_aux_loss += aux_loss.item()
            mean_aux_accuracy += aux_accuracy.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "aux": mean_aux_loss / num_updates,
            "aux_variant_accuracy": mean_aux_accuracy / num_updates,
        }

    def compute_returns(self, obs):
        # Upstream evaluates the next observation to bootstrap GAE. The runner
        # feeds that same observation to act() at the start of its next rollout;
        # retaining the bootstrap's RNN state would consume it twice in the critic.
        if not self.critic.is_recurrent:
            return super().compute_returns(obs)
        hidden = self.critic.get_hidden_state()
        if isinstance(hidden, tuple):
            hidden = tuple(state.clone() for state in hidden)
        elif hidden is not None:
            hidden = hidden.clone()
        try:
            return super().compute_returns(obs)
        finally:
            self.critic.reset(hidden_state=hidden)

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        # This small construction adapter targets the pinned RSL-RL 5.2 contract.
        # RND, symmetry and distributed execution are deliberately not configured.
        # RSL's runner/logger expects these keys, normally inserted by its default
        # construction path even when the optional mechanisms are disabled.
        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["algorithm"].setdefault("symmetry_cfg", None)
        algorithm = dict(cfg["algorithm"])
        algorithm.pop("class_name")
        algorithm.pop("share_cnn_encoders", None)
        groups = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        models = []
        for name, output in (("actor", env.num_actions), ("critic", 1)):
            # RSL constructors pop nested distribution keys; preserve the saved
            # runner configuration for checkpoint reconstruction and logging.
            options = deepcopy(cfg[name])
            cls = resolve_callable(options.pop("class_name"))
            models.append(cls(obs, groups, name, output, **options).to(device))
        storage = CompactStorage(
            env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        return CompactPPO(
            *models, storage, device=device, multi_gpu_cfg=cfg["multi_gpu"], **algorithm
        )


def runner_cfg(experiment, iterations=3000):
    cfg = RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name="active_perception_arms.policy:"
            + ("VisualMemoryModel" if experiment.memory == "gru" else "VisualFeedForwardModel"),
            hidden_dims=(128, 128),
            obs_normalization=True,
            rnn_type="gru",
            rnn_hidden_dim=128,
            cnn_cfg={
                "output_channels": [16, 32],
                "kernel_size": [5, 3],
                "stride": [2, 2],
                "padding": "zeros",
                "activation": "elu",
                "max_pool": False,
                "global_pool": "none",
                "spatial_softmax": True,
            },
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.4,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            class_name="RNNModel",
            hidden_dims=(128, 128),
            obs_normalization=True,
            rnn_type="gru",
            rnn_hidden_dim=128,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            class_name="active_perception_arms.policy:CompactPPO",
            num_learning_epochs=4,
            num_mini_batches=8,
            learning_rate=3e-4,
            entropy_coef=0.003,
        ),
        obs_groups={"actor": ("proprio", *experiment.sensors), "critic": ("critic",)},
        seed=experiment.seed,
        num_steps_per_env=24,
        max_iterations=iterations,
        experiment_name=f"{experiment.task}_{experiment.condition}",
        logger="tensorboard",
        upload_model=False,
        save_interval=100,
    )
    if experiment.policy_profile == "legacy_ff":
        # Match the recovered legacy widths and PPO settings, while retaining
        # current RGB encoders, physical scoring, geometry and observation schema.
        cfg.actor.hidden_dims = cfg.critic.hidden_dims = (256, 256, 128)
        cfg.critic.class_name = "MLPModel"
        cfg.critic.rnn_type = cfg.critic.rnn_hidden_dim = None
        cfg.critic.rnn_num_layers = None
        cfg.actor.distribution_cfg["init_std"] = 1.0
        cfg.algorithm.learning_rate = 0.001
        cfg.algorithm.entropy_coef = 0.005
        cfg.algorithm.num_learning_epochs = 5
        cfg.algorithm.num_mini_batches = 4
    if experiment.actor_observation == "privileged":
        cfg.obs_groups = {"actor": ("privileged",), "critic": ("privileged",)}
        cfg.actor.cnn_cfg = None
        for model in (cfg.actor, cfg.critic):
            model.class_name = "RNNModel" if experiment.memory == "gru" else "MLPModel"
            model.hidden_dims = (256, 256)
            if experiment.memory == "none":
                model.rnn_type = model.rnn_hidden_dim = model.rnn_num_layers = None
    return cfg


def runner_dict(runner_options, experiment, aux_coef=0.0):
    """Serialisable runner config; adds auxiliary-loss options absent from MjLab's dataclasses."""
    from dataclasses import asdict

    cfg = asdict(runner_options)
    if getattr(experiment, "aux_targets", False):
        # Variant classes plus goal-minus-TCP (3) and socket XY (2) regressions.
        cfg["actor"]["aux_dim"] = AUX_CLASSES + 5
        cfg["algorithm"]["aux_coef"] = float(aux_coef)
    elif aux_coef:
        raise ValueError("aux_coef requires aux_targets in the experiment")
    return cfg
