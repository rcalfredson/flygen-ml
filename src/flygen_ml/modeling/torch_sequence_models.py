from __future__ import annotations

from typing import Any

import numpy as np

from flygen_ml.modeling.sequence_models import FlySequenceExample


def _import_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for model_kind 'sequence_conv1d_meanpool_torch_v1'. "
            "Install torch in the active Conda environment before training this model."
        ) from exc
    return torch, nn


def _segment_cap_from_config(
    config: dict[str, object],
    *,
    key: str,
    fallback_key: str | None = None,
    default: int | None = None,
) -> int | None:
    if key in config:
        raw_value = config[key]
    elif fallback_key is not None and fallback_key in config:
        raw_value = config[fallback_key]
    else:
        raw_value = default
    if raw_value is None:
        return None
    value = int(raw_value)
    return value if value > 0 else None


def _evenly_spaced_indices(indices: np.ndarray, max_count: int | None) -> np.ndarray:
    if max_count is None or max_count <= 0 or len(indices) <= max_count:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=max_count).round().astype(int)
    return indices[positions]


def _random_sample_indices(
    indices: np.ndarray,
    max_count: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_count is None or max_count <= 0 or len(indices) <= max_count:
        return indices
    sampled = rng.choice(indices, size=max_count, replace=False)
    return np.sort(sampled)


def _fit_channel_scaler(
    x: np.ndarray,
    examples: list[FlySequenceExample],
    max_segments_per_fly: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.concatenate(
        [_evenly_spaced_indices(example.segment_indices, max_segments_per_fly) for example in examples]
    )
    selected_x = x[selected]
    means = selected_x.mean(axis=(0, 1))
    stds = selected_x.std(axis=(0, 1))
    stds = np.where(stds > 1e-6, stds, 1.0)
    return means.astype(np.float32), stds.astype(np.float32)


def _scaled_sequences(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return ((x - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)).astype(np.float32)


def _resolve_device(torch: Any, config: dict[str, object]):
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested device 'cuda' but torch.cuda.is_available() is false")
    return device


def _build_module(
    *,
    n_channels: int,
    conv_channels: int,
    embedding_dim: int,
    n_genotype_classes: int,
    n_cohort_classes: int,
    dropout: float,
):
    torch, nn = _import_torch()

    class SegmentConvMeanPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.segment_encoder = nn.Sequential(
                nn.Conv1d(n_channels, conv_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(conv_channels * 2, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.genotype_head = nn.Linear(embedding_dim, n_genotype_classes)
            self.cohort_head = nn.Linear(embedding_dim, n_cohort_classes)

        def encode_segments(self, segments):
            return self.projection(self.segment_encoder(segments))

        def forward_fly(self, segments):
            fly_embedding = self.encode_segments(segments).mean(dim=0)
            return self.genotype_head(fly_embedding), self.cohort_head(fly_embedding)

    return SegmentConvMeanPool()


def _state_dict_to_jsonable(module) -> dict[str, object]:
    return {
        key: value.detach().cpu().numpy().tolist()
        for key, value in module.state_dict().items()
    }


def train_torch_sequence_meanpool(
    x: np.ndarray,
    examples: list[FlySequenceExample],
    *,
    config: dict[str, object],
) -> dict[str, object]:
    if not examples:
        raise ValueError("cannot train torch sequence model with no fly examples")
    torch, nn = _import_torch()
    random_seed = int(config.get("random_seed", 0))
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    max_iter = int(config.get("max_iter", 50))
    learning_rate = float(config.get("learning_rate", 0.001))
    weight_decay = float(config.get("weight_decay", config.get("l2_reg", 0.0001)))
    conv_channels = int(config.get("conv_channels", 32))
    embedding_dim = int(config.get("embedding_dim", config.get("hidden_dim", 64)))
    dropout = float(config.get("dropout", 0.1))
    cohort_loss_weight = float(config.get("cohort_loss_weight", 1.0))
    train_max_segments_per_fly = _segment_cap_from_config(
        config,
        key="train_max_segments_per_fly",
        fallback_key="max_segments_per_fly",
        default=200,
    )
    eval_max_segments_per_fly = _segment_cap_from_config(
        config,
        key="eval_max_segments_per_fly",
        fallback_key="max_segments_per_fly",
        default=0,
    )
    scaler_max_segments_per_fly = _segment_cap_from_config(
        config,
        key="scaler_max_segments_per_fly",
        fallback_key="train_max_segments_per_fly",
        default=train_max_segments_per_fly,
    )
    genotype_labels = sorted({example.genotype for example in examples})
    cohort_labels = sorted({example.cohort for example in examples})
    if len(genotype_labels) < 2:
        raise ValueError(f"expected at least two genotype labels, got {genotype_labels}")
    if len(cohort_labels) < 2:
        raise ValueError(f"expected at least two cohort labels, got {cohort_labels}")
    genotype_to_index = {label: idx for idx, label in enumerate(genotype_labels)}
    cohort_to_index = {label: idx for idx, label in enumerate(cohort_labels)}

    means, stds = _fit_channel_scaler(x, examples, scaler_max_segments_per_fly)
    x_scaled = _scaled_sequences(x, means, stds)
    device = _resolve_device(torch, config)
    module = _build_module(
        n_channels=x.shape[2],
        conv_channels=conv_channels,
        embedding_dim=embedding_dim,
        n_genotype_classes=len(genotype_labels),
        n_cohort_classes=len(cohort_labels),
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(random_seed + 1)

    module.train()
    for _ in range(max_iter):
        optimizer.zero_grad()
        for example in examples:
            indices = _random_sample_indices(example.segment_indices, train_max_segments_per_fly, rng)
            segment_batch = torch.as_tensor(x_scaled[indices], dtype=torch.float32, device=device).permute(0, 2, 1)
            genotype_logits, cohort_logits = module.forward_fly(segment_batch)
            genotype_target = torch.tensor([genotype_to_index[example.genotype]], dtype=torch.long, device=device)
            cohort_target = torch.tensor([cohort_to_index[example.cohort]], dtype=torch.long, device=device)
            loss = criterion(genotype_logits.unsqueeze(0), genotype_target)
            loss = loss + cohort_loss_weight * criterion(cohort_logits.unsqueeze(0), cohort_target)
            (loss / len(examples)).backward()
        optimizer.step()

    return {
        "model_kind": "sequence_conv1d_meanpool_torch_v1",
        "module": module,
        "state_dict": _state_dict_to_jsonable(module),
        "genotype_labels": genotype_labels,
        "cohort_labels": cohort_labels,
        "input_means": means,
        "input_stds": stds,
        "n_channels": x.shape[2],
        "conv_channels": conv_channels,
        "embedding_dim": embedding_dim,
        "dropout": dropout,
        "cohort_loss_weight": cohort_loss_weight,
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": str(device),
        "train_max_segments_per_fly": 0 if train_max_segments_per_fly is None else train_max_segments_per_fly,
        "eval_max_segments_per_fly": 0 if eval_max_segments_per_fly is None else eval_max_segments_per_fly,
        "scaler_max_segments_per_fly": 0 if scaler_max_segments_per_fly is None else scaler_max_segments_per_fly,
        "segment_sampling": "random_without_replacement_per_epoch",
    }


def predict_torch_sequence_meanpool(
    x: np.ndarray,
    examples: list[FlySequenceExample],
    *,
    model: dict[str, object],
) -> list[dict[str, object]]:
    torch, _ = _import_torch()
    module = model["module"]
    device = torch.device(str(model.get("device", "cpu")))
    module.eval()
    means = np.asarray(model["input_means"], dtype=np.float32)
    stds = np.asarray(model["input_stds"], dtype=np.float32)
    x_scaled = _scaled_sequences(x, means, stds)
    max_segments_per_fly = int(model.get("eval_max_segments_per_fly", 0)) or None
    genotype_labels = [str(label) for label in model["genotype_labels"]]
    cohort_labels = [str(label) for label in model["cohort_labels"]]
    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for example in examples:
            indices = _evenly_spaced_indices(example.segment_indices, max_segments_per_fly)
            segment_batch = torch.as_tensor(x_scaled[indices], dtype=torch.float32, device=device).permute(0, 2, 1)
            genotype_logits, cohort_logits = module.forward_fly(segment_batch)
            genotype_probs = torch.softmax(genotype_logits, dim=0).detach().cpu().numpy()
            cohort_probs = torch.softmax(cohort_logits, dim=0).detach().cpu().numpy()
            genotype_idx = int(genotype_probs.argmax())
            cohort_idx = int(cohort_probs.argmax())
            predictions.append(
                {
                    "fly_id": example.fly_id,
                    "sample_key": example.sample_key,
                    "actual_genotype": example.genotype,
                    "predicted_genotype": genotype_labels[genotype_idx],
                    "genotype_probability": float(genotype_probs[genotype_idx]),
                    "actual_cohort": example.cohort,
                    "predicted_cohort": cohort_labels[cohort_idx],
                    "cohort_probability": float(cohort_probs[cohort_idx]),
                    "n_segments": example.n_segments,
                    "n_segments_with_qc_flags": example.n_segments_with_qc_flags,
                }
            )
    return predictions


def serializable_torch_sequence_model(model: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in model.items():
        if key == "module":
            continue
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        else:
            payload[key] = value
    return payload
