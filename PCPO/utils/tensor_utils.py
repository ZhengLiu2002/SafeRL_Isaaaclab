import torch


def to_device_tensor(data, device, dtype=None):
    """Convert input to a torch tensor on the target device."""
    if isinstance(data, torch.Tensor):
        if dtype is not None and data.dtype != dtype:
            data = data.to(dtype)
        return data.to(device)
    return torch.as_tensor(data, device=device, dtype=dtype)


def extract_obs(obs, device, obs_key="policy"):
    """Return the policy observation tensor (handles dict observations)."""
    if isinstance(obs, dict):
        # Prefer the policy key; otherwise pick the first available tensor.
        obs_val = obs.get(obs_key, next(iter(obs.values())))
    else:
        obs_val = obs
    return to_device_tensor(obs_val, device=device)


def extract_costs(info, cost_keys, device, num_envs):
    """Collect per-env costs from Isaac Lab info/log fields."""
    if cost_keys is None:
        cost_keys = []
    costs = torch.zeros(num_envs, device=device)
    if not cost_keys:
        return costs

    log_src = None
    if isinstance(info, dict):
        if "log" in info:
            log_src = info["log"]
        elif "extras" in info and isinstance(info["extras"], dict) and "log" in info["extras"]:
            log_src = info["extras"]["log"]
    if log_src is None:
        return costs

    for key in cost_keys:
        if key in log_src:
            # We assume costs are positive, but rewards/penalties in Isaac Lab are often negative.
            # We take the absolute value to ensure costs are positive constraints.
            costs += to_device_tensor(log_src[key], device=device).abs()
    return costs
