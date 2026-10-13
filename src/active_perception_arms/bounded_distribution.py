"""Gaussian exploration with an explicit penalty for unusable out-of-range means.

The environment clips requests to [-1,1]. RSL exposes no auxiliary actor-loss
callback; the hook adds the exact gradient of coefficient*mean(excess**2) to
its existing PPO backward pass, without changing the forward distribution.
It is inactive in no-grad rollout/evaluation. PPO's logged loss excludes this
auxiliary term; last_mean_penalty records it separately.
"""

import torch
from rsl_rl.modules.distribution import GaussianDistribution


class SquashedGaussian(GaussianDistribution):
    """Tanh actions with corrected density/entropy and bounded latent means.

    KL remains the base Gaussian KL under the common bijective transform.
    Entropy uses a reparameterized sample of the executed bounded distribution.
    """

    @staticmethod
    def _mean(output):
        return 2 * torch.tanh(output / 2)

    def update(self, mlp_output):
        super().update(self._mean(mlp_output))

    def sample(self):
        return self._distribution.sample().tanh()

    def deterministic_output(self, mlp_output):
        return self._mean(mlp_output).tanh()

    def log_prob(self, outputs):
        bounded = outputs.clamp(-1 + 1e-6, 1 - 1e-6)
        return (
            self._distribution.log_prob(torch.atanh(bounded)) - torch.log1p(-bounded.square())
        ).sum(-1)

    @property
    def entropy(self):
        z = self._distribution.rsample()
        correction = 2 * (torch.log(z.new_tensor(2.0)) - z - torch.nn.functional.softplus(-2 * z))
        return self._distribution.entropy().sum(-1) + correction.sum(-1)

    def as_deterministic_output_module(self):
        raise NotImplementedError("Use the checkpoint evaluator for squashed policies")


class RegularizedGaussian(GaussianDistribution):
    def __init__(self, *args, mean_penalty=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_penalty = mean_penalty
        self.last_mean_penalty = 0.0

    def update(self, mlp_output):
        if mlp_output.requires_grad and self.mean_penalty:
            excess = mlp_output.detach() - mlp_output.detach().clamp(-1, 1)
            additional = 2 * self.mean_penalty * excess / mlp_output.numel()
            mlp_output.register_hook(lambda gradient: gradient + additional)
            self.last_mean_penalty = float(self.mean_penalty * excess.square().mean())
        super().update(mlp_output)
