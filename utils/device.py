import subprocess

import torch


def query_gpu_memory():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index, total, used, free = map(int, parts)
        except ValueError:
            continue
        gpus.append({"index": index, "total_mb": total, "used_mb": used, "free_mb": free})
    return gpus


def select_device(device_arg="auto", min_free_gb=6.0, reserve_gb=4.0, verbose=True):
    if device_arg == "cpu":
        if verbose:
            print("Device: cpu")
        return torch.device("cpu")

    if device_arg.startswith("cuda:"):
        if not torch.cuda.is_available():
            if verbose:
                print("CUDA requested but unavailable; falling back to cpu")
            return torch.device("cpu")
        idx = int(device_arg.split(":", 1)[1])
        gpus = {gpu["index"]: gpu for gpu in query_gpu_memory()}
        gpu = gpus.get(idx)
        if gpu is not None:
            usable_gb = max((gpu["free_mb"] / 1024.0) - reserve_gb, 0.0)
            if verbose:
                print(
                    f"Device: cuda:{idx} | free={gpu['free_mb']/1024.0:.1f}GB "
                    f"reserve={reserve_gb:.1f}GB usable~={usable_gb:.1f}GB"
                )
            if usable_gb < min_free_gb:
                raise RuntimeError(
                    f"cuda:{idx} has only {usable_gb:.1f}GB usable after reserve; "
                    f"need at least {min_free_gb:.1f}GB. Choose another GPU or lower --min_free_gb."
                )
        torch.cuda.set_device(idx)
        return torch.device(device_arg)

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            if verbose:
                print("CUDA unavailable; falling back to cpu")
            return torch.device("cpu")
        if verbose:
            print("Device: cuda")
        return torch.device("cuda")

    if device_arg != "auto":
        raise ValueError(f"Unsupported --device value: {device_arg}")

    if not torch.cuda.is_available():
        if verbose:
            print("CUDA unavailable; falling back to cpu")
        return torch.device("cpu")

    gpus = query_gpu_memory()
    if not gpus:
        if verbose:
            print("Could not query nvidia-smi; using cuda:0")
        torch.cuda.set_device(0)
        return torch.device("cuda:0")

    candidates = []
    for gpu in gpus:
        free_gb = gpu["free_mb"] / 1024.0
        usable_gb = max(free_gb - reserve_gb, 0.0)
        if usable_gb >= min_free_gb:
            candidates.append((usable_gb, gpu))
    if not candidates:
        summary = "; ".join(f"cuda:{gpu['index']} free={gpu['free_mb']/1024.0:.1f}GB" for gpu in gpus)
        raise RuntimeError(
            f"No GPU has enough free memory after reserving {reserve_gb:.1f}GB. "
            f"Need usable >= {min_free_gb:.1f}GB. Current: {summary}. "
            "Use --device cpu, --device cuda:N, or lower --min_free_gb if appropriate."
        )

    _, best = max(candidates, key=lambda item: (item[0], item[1]["free_mb"]))
    free_gb = best["free_mb"] / 1024.0
    usable_gb = max(free_gb - reserve_gb, 0.0)
    if verbose:
        print(
            f"Auto-selected device: cuda:{best['index']} | "
            f"total={best['total_mb']/1024.0:.1f}GB used={best['used_mb']/1024.0:.1f}GB "
            f"free={free_gb:.1f}GB reserve={reserve_gb:.1f}GB usable~={usable_gb:.1f}GB"
        )
    torch.cuda.set_device(best["index"])
    return torch.device(f"cuda:{best['index']}")
