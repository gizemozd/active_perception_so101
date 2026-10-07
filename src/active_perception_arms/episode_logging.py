"""Bridge MjLab's empty per-step logs to the pinned RSL episode aggregator."""


def install_episode_logging(logger):
    """Retain episode metrics even when a rollout begins before any episode ends."""
    original = logger.process_env_step

    def process(rewards, dones, extras, intrinsic_rewards=None):
        # RSL 5.2 selects metric keys from ep_extras[0]. MjLab emits log={}
        # until a reset; retaining that empty entry silently drops later metrics.
        filtered = {
            key: value for key, value in extras.items() if key not in ("episode", "log") or value
        }
        return original(rewards, dones, filtered, intrinsic_rewards)

    logger.process_env_step = process
