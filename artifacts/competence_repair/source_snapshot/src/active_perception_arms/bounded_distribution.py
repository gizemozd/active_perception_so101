"""Gaussian exploration with an explicit penalty for unusable out-of-range means.

The environment clips requests to [-1,1]. RSL exposes no auxiliary actor-loss
callback; the hook adds the exact gradient of coefficient*mean(excess**2) to
its existing PPO backward pass, without changing the forward distribution.
It is inactive in no-grad rollout/evaluation. PPO's logged loss excludes this
auxiliary term; last_mean_penalty records it separately.
"""

from rsl_rl.modules.distribution import GaussianDistribution


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
