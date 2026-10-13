import torch
from rsl_rl.modules.distribution import GaussianDistribution

from active_perception_arms.bounded_distribution import RegularizedGaussian


def test_auxiliary_gradient_equals_explicit_mean_excess_loss():
    means = torch.tensor([[5.0, -0.2, -3.0], [0.1, 2.0, -1.0]], requires_grad=True)
    reference = means.detach().clone().requires_grad_(True)
    a = RegularizedGaussian(3, init_std=0.3, mean_penalty=0.02)
    b = GaussianDistribution(3, init_std=0.3)
    action = torch.zeros_like(means)
    a.update(means)
    b.update(reference)
    torch.testing.assert_close(a.mean, b.mean)
    (-a.log_prob(action).mean()).backward()
    loss = -b.log_prob(action).mean() + 0.02 * (reference - reference.clamp(-1, 1)).square().mean()
    loss.backward()
    torch.testing.assert_close(means.grad, reference.grad)
    torch.testing.assert_close(a.std_param.grad, b.std_param.grad)
    with torch.no_grad():
        a.update(means.detach())
        torch.testing.assert_close(a.deterministic_output(means), means)
