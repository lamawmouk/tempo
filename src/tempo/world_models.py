"""Frozen world models: loading, image transform, per-frame latents for F_phi, and the DINO-WM specifics.

LeWM and PLDM have one latent vector per frame: z = encode({'pixels': frame})['emb'][:, 0].
DINO-WM (stable-worldmodel's PreJEPA) has a patch grid plus optional extra state encoders; F_phi then sees
[patch-mean of the pixel grid, extra embeddings], and the planner's own cost compares the parts.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def image_transform(size: int = 224) -> Callable:
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(**IMAGENET), T.Resize(size=size)])


def load_world_model(name: str, device: str):
    """Load a checkpoint from $STABLEWM_HOME/checkpoints (stable-worldmodel format), frozen and in eval mode."""
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(name).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def is_dinowm(model) -> bool:
    return type(model).__name__.lower().startswith("prejepa")


def dinowm_emb_keys(model) -> list:
    """Extra encoders that carry state (all but `action`), in the order PreJEPA concatenates them."""
    return [k for k in model.extra_encoders if k != "action"]


def pool_grid(emb: torch.Tensor) -> torch.Tensor:
    """(..., n_patches, D) -> (..., D): mean over the patch axis (always dim -2)."""
    assert emb.dim() >= 3, f"expected (..., n_patches, D), got {tuple(emb.shape)}"
    return emb.mean(dim=-2)


def dinowm_pool(out: dict, keys: Sequence[str], target: str = "emb") -> torch.Tensor:
    """[patch-mean pixel embedding, extra embeddings] from PreJEPA's per-source outputs. The fused `emb` is not used: after a
    rollout it also carries the tiled action embedding."""
    parts = [pool_grid(out[f"pixels_{target}"])] + [out[f"{k}_{target}"] for k in keys]
    return torch.cat([x.float() for x in parts], dim=-1)


def _frames_chw(x, transform) -> torch.Tensor:
    x = np.asarray(x) if not torch.is_tensor(x) else x.cpu().numpy()
    frames = x[:, -1] if x.ndim == 5 else x  # (B, T, H, W, C) -> current frame
    return torch.stack([transform(tv_tensors.Image(f)) for f in np.transpose(frames, (0, 3, 1, 2))])


def state_encoder(model, transform: Callable, device: str, process: Optional[Dict] = None) -> Callable[[dict], torch.Tensor]:
    """fn(info) -> (B, D) latent of the current frame, in the space F_phi is trained on."""
    keys = dinowm_emb_keys(model) if is_dinowm(model) else []
    process = process or {}

    @torch.no_grad()
    def fn(info: dict) -> torch.Tensor:
        px = _frames_chw(info["pixels"], transform).to(device)[:, None]
        if not keys:
            return model.encode({"pixels": px})["emb"][:, 0].float().cpu()
        payload = {"pixels": px}
        for k in keys:  # PreJEPA does not normalise its extra inputs: apply the training-set scaler
            v = info[k]
            v = np.asarray(v.cpu().numpy() if torch.is_tensor(v) else v, np.float32)
            v = (v[:, -1] if v.ndim == 3 else v).reshape(px.shape[0], -1)
            if k in process:
                v = process[k].transform(v)
            payload[k] = torch.as_tensor(np.asarray(v, np.float32), device=device)[:, None]
        return dinowm_pool(model.encode(payload, emb_keys=keys), keys)[:, 0].cpu()

    return fn


class ChunkedRollout:
    """Proxy around a PreJEPA model that runs `rollout` in chunks along the candidate axis.

    50 envs x 300 candidates of (context + horizon) x 256 tokens does not fit in one predictor call. Chunking is exact:
    every candidate is rolled out independently from the same encoded context. Everything else is forwarded."""

    def __init__(self, model, max_sequences: int = 1500):
        object.__setattr__(self, "_m", model)
        object.__setattr__(self, "max_sequences", int(max_sequences))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_m"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_m"), name, value)

    def __call__(self, *a, **k):
        return object.__getattribute__(self, "_m")(*a, **k)

    def rollout(self, info: dict, action_sequence: torch.Tensor):
        m = object.__getattribute__(self, "_m")
        B, S = action_sequence.shape[:2]
        step = max(1, object.__getattribute__(self, "max_sequences") // max(B, 1))
        if S <= step:
            return m.rollout(info, action_sequence)
        has_S = {k for k, v in info.items() if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == B and v.shape[1] == S}
        outs = [
            m.rollout({k: (v[:, s : s + step] if k in has_S else v) for k, v in info.items()}, action_sequence[:, s : s + step])
            for s in range(0, S, step)
        ]
        merged = dict(info)
        for k, v0 in outs[0].items():
            if torch.is_tensor(v0) and v0.dim() >= 2 and v0.shape[0] == B and (k in has_S or k not in info or v0 is not info.get(k)):
                parts = [o[k] for o in outs]
                if all(torch.is_tensor(x) and x.dim() >= 2 for x in parts) and sum(int(x.shape[1]) for x in parts) == S:
                    merged[k] = torch.cat(parts, dim=1)
                    continue
            merged[k] = v0
        return merged
