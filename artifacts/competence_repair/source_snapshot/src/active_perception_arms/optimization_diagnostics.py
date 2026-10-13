"""Read-only diagnostics for search runs on the pinned RSL 5.2 PPO loop."""

import torch


def install_optimization_diagnostics(algorithm, logger):
    """Observe native minibatches without changing losses, gradients or RNG use."""
    original_generator = algorithm.storage.recurrent_mini_batch_generator
    original_log_prob = algorithm.actor.get_output_log_prob
    original_update = algorithm.update
    original_step = logger.process_env_step
    state = {"batch": None, "ppo": [], "episodes": [], "saturation": []}
    original_act = algorithm.act

    def act(obs):
        actions = original_act(obs)
        state["saturation"].append((actions.detach().abs() >= 1).float().mean())
        return actions

    def generator(*args, **kwargs):
        try:
            for batch in original_generator(*args, **kwargs):
                state["batch"] = batch
                yield batch
        finally:
            state["batch"] = None

    def log_prob(actions):
        result = original_log_prob(actions)
        batch = state["batch"]
        if batch is not None:
            with torch.no_grad():
                ratio = (result.detach() - batch.old_actions_log_prob.squeeze()).exp()
                clip = (torch.abs(ratio - 1) > algorithm.clip_param).float().mean()
                size = batch.observations.batch_size[0]
                params = tuple(
                    p[:size].detach() for p in algorithm.actor.output_distribution_params
                )
                kl = algorithm.actor.get_kl_divergence(batch.old_distribution_params, params).mean()
                state["ppo"].append(torch.stack((kl, clip)))
        return result

    def process(rewards, dones, extras, intrinsic_rewards=None):
        metrics = extras.get("log", {})
        success = metrics.get("Episode_Metrics/success_rate")
        if success is not None:
            # MjLab's last-value metric is the mean over precisely the reset envs.
            # Weight it by completed episodes instead of averaging reset batches.
            count = dones.detach().count_nonzero().float()
            value = torch.as_tensor(success, device=count.device).detach()
            state["episodes"].append(torch.stack((count * value, count)))
        return original_step(rewards, dones, extras, intrinsic_rewards)

    def update():
        result = original_update()
        if state["ppo"]:
            kl, clip = torch.stack(state["ppo"]).mean(0).tolist()
            result.update(diagnostic_kl=kl, diagnostic_clip_fraction=clip)
        if state["episodes"]:
            successes, completed = torch.stack(state["episodes"]).sum(0).tolist()
            result["diagnostic_completed_episodes"] = completed
            result["diagnostic_successes"] = successes
            if completed:
                result["diagnostic_episode_weighted_success"] = successes / completed
        else:
            result["diagnostic_completed_episodes"] = 0
            result["diagnostic_successes"] = 0
        if state["saturation"]:
            result["diagnostic_action_saturation"] = torch.stack(state["saturation"]).mean().item()
        distribution = algorithm.actor.distribution
        std = distribution.std_param.detach().clamp(*distribution.std_range)
        result.update({f"diagnostic_action_std_{i}": v for i, v in enumerate(std.tolist())})
        for key in ("ppo", "episodes", "saturation"):
            state[key].clear()
        return result

    algorithm.storage.recurrent_mini_batch_generator = generator
    algorithm.actor.get_output_log_prob = log_prob
    algorithm.act = act
    algorithm.update = update
    logger.process_env_step = process
