# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Record actual exported tensor bytes before the rollout engine reloads them."""

import hashlib
import json
import os
from pathlib import Path


def record_weight_update(args, version_dir, rollout_engines):
    import torch
    import torch.distributed as dist
    from safetensors import safe_open

    if dist.is_initialized() and dist.get_rank() != 0:
        return
    root = Path(version_dir)
    name = "model.layers.0.mlp.down_proj.weight"
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        files = [root / mapping[name]]
    else:
        files = sorted(root.glob("*.safetensors"))
    digest = None
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as archive:
            if name in archive.keys():
                tensor = archive.get_tensor(name).contiguous()
                digest = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
                break
    if digest is None:
        raise RuntimeError(f"weight update audit could not find {name} in {root}")
    output = Path(os.environ["RL_KERNEL_WEIGHT_AUDIT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "version": root.name,
        "tensor": name,
        "sha256": digest,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    (output / f"{root.name}.json").write_text(json.dumps(record, indent=2) + "\n")
