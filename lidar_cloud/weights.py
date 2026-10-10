"""Parameter-only weights; initialization helpers for other algorithms."""

import hashlib
import json
from pathlib import Path

import torch

from .models.network import ZMultiTaskNetL

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_manifest():
    return json.loads((RESOURCE_DIR / "models.json").read_text(encoding="utf-8"))


def weight_path(model):
    entry = model_manifest()[model.upper()]
    return RESOURCE_DIR / entry["file"]


def load_model(model="E3", device="cpu", checkpoint=None):
    """Load E1--E4 for detection, or R1/R2 for reuse (no optimizer)."""
    model = model.upper()
    if model not in {"E1", "E2", "E3", "E4", "R1", "R2"}:
        raise ValueError("Choose E1, E2, E3, E4, R1, or R2. E5 is a rule-based algorithm.")
    entry = model_manifest()[model]
    path = Path(checkpoint) if checkpoint is not None else weight_path(model)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights missing: {path}")
    if checkpoint is None and sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Weight checksum mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    network = ZMultiTaskNetL(n_recon_ch=5 if model == "R1" else 0, dropout=0.1)
    network.load_state_dict(state, strict=True)
    return network.to(device).eval()


def load_backbone(model="R2", device="cpu"):
    """Return the pretrained temporal-vertical backbone for a custom head."""
    return load_model(model, device).backbone
