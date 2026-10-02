"""Shared recurrent visual actor and compact PPO storage for RSL-RL 5.2."""

import torch
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
from rsl_rl.algorithms import PPO
from rsl_rl.modules import RNN
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class VisualMemoryModel(SpatialSoftmaxCNNModel):
    """Encode views and proprioception once; GRU state is shared by both arm outputs."""

    is_recurrent = True

    def __init__(self, *args, rnn_type="gru", rnn_hidden_dim=128, rnn_num_layers=1, **kwargs):
        self.latent_dim = rnn_hidden_dim
        super().__init__(*args, **kwargs)
        self.rnn = RNN(super()._get_latent_dim(), rnn_hidden_dim, rnn_num_layers, rnn_type)

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
        return self.rnn(self.encode(obs, masks), masks, hidden_state).squeeze(0)

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

    def __init__(self, *args, rnn_type=None, rnn_hidden_dim=None, rnn_num_layers=None, **kwargs):
        super().__init__(*args, **kwargs)

    def get_latent(self, obs, masks=None, hidden_state=None):
        # MLPModel.forward has already removed recurrent padding for this model.
        return self.encode(obs)


class CompactPPO(PPO):
    """Unchanged PPO updates; only model/storage construction is specialized."""

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
            options = dict(cfg[name])
            cls = resolve_callable(options.pop("class_name"))
            models.append(cls(obs, groups, name, output, **options).to(device))
        storage = CompactStorage(
            env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        return CompactPPO(
            *models, storage, device=device, multi_gpu_cfg=cfg["multi_gpu"], **algorithm
        )


def runner_cfg(experiment, iterations=3000):
    return RslRlOnPolicyRunnerCfg(
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
