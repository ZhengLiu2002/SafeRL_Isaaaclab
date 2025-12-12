import torch


def sanitize_term_key(key: str) -> str:
    name = str(key)
    if "/" in name:
        name = name.split("/")[-1]
    return name.replace(" ", "_")


def extract_log_dict(info):
    if isinstance(info, dict):
        if "log" in info:
            return info["log"]
        extras = info.get("extras", None)
        if isinstance(extras, dict) and "log" in extras:
            return extras["log"]
        return None

    if isinstance(info, (list, tuple)):
        merged = {}
        for entry in info:
            entry_log = extract_log_dict(entry)
            if entry_log is None:
                continue
            for k, v in entry_log.items():
                merged.setdefault(k, []).append(v)
        if not merged:
            return None
        stacked = {}
        for k, vals in merged.items():
            try:
                tensors = [torch.as_tensor(v) for v in vals]
                stacked[k] = torch.stack(tensors)
            except Exception:
                stacked[k] = vals
        return stacked

    return None


def to_device_tensor(data, device, dtype=None):
    if isinstance(data, torch.Tensor):
        if dtype is not None and data.dtype != dtype:
            data = data.to(dtype)
        return data.to(device)
    return torch.as_tensor(data, device=device, dtype=dtype)


def extract_obs(obs, device, obs_key="policy"):
    if isinstance(obs, dict):
        obs_val = obs.get(obs_key, next(iter(obs.values())))
    else:
        obs_val = obs
    return to_device_tensor(obs_val, device=device)


def extract_costs(info, cost_keys, device, num_envs):
    if cost_keys is None:
        cost_keys = []
    costs = torch.zeros(num_envs, device=device)
    if not cost_keys:
        return costs

    log_src = extract_log_dict(info)
    if log_src is None:
        return costs

    normalized_cost_keys = {sanitize_term_key(k).lower() for k in cost_keys}

    for key, raw_val in log_src.items():
        name_lower = sanitize_term_key(key).lower()
        raw_key_lower = str(key).lower()
        if name_lower not in normalized_cost_keys and raw_key_lower not in normalized_cost_keys:
            continue

        tensor_val = to_device_tensor(raw_val, device=device)
        flat = tensor_val.view(-1)
        if flat.numel() == 1:
            costs += flat.expand_as(costs).abs()
        else:
            costs += flat[: num_envs].abs()
    return costs
