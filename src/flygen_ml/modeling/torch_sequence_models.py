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


def _random_contiguous_indices(
    indices: np.ndarray,
    max_count: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_count is None or max_count <= 0 or len(indices) <= max_count:
        return indices
    start = int(rng.integers(0, len(indices) - max_count + 1))
    return indices[start : start + max_count]


def _centered_contiguous_indices(indices: np.ndarray, max_count: int | None) -> np.ndarray:
    if max_count is None or max_count <= 0 or len(indices) <= max_count:
        return indices
    start = (len(indices) - max_count) // 2
    return indices[start : start + max_count]


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


def _fit_side_scaler(
    examples: list[FlySequenceExample],
    side_inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([side_inputs[example.fly_id] for example in examples]).astype(np.float32)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds = np.where(stds > 1e-6, stds, 1.0)
    return means.astype(np.float32), stds.astype(np.float32)


def _scaled_side_input(
    example: FlySequenceExample,
    side_inputs: dict[str, np.ndarray] | None,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray | None:
    if side_inputs is None:
        return None
    return ((side_inputs[example.fly_id] - means) / stds).astype(np.float32)


def _balanced_accuracy(actual: list[str], predicted: list[str]) -> float:
    labels = sorted(set(actual))
    if not labels:
        return 0.0
    recalls = []
    for label in labels:
        label_total = sum(value == label for value in actual)
        label_correct = sum(
            actual_value == label and predicted_value == label
            for actual_value, predicted_value in zip(actual, predicted, strict=True)
        )
        recalls.append(label_correct / label_total if label_total else 0.0)
    return float(np.mean(recalls))


def _prediction_metrics(predictions: list[dict[str, object]]) -> dict[str, float]:
    if not predictions:
        return {
            "joint_accuracy": 0.0,
            "genotype_accuracy": 0.0,
            "genotype_balanced_accuracy": 0.0,
            "cohort_accuracy": 0.0,
            "cohort_balanced_accuracy": 0.0,
        }
    actual_genotypes = [str(row["actual_genotype"]) for row in predictions]
    predicted_genotypes = [str(row["predicted_genotype"]) for row in predictions]
    actual_cohorts = [str(row["actual_cohort"]) for row in predictions]
    predicted_cohorts = [str(row["predicted_cohort"]) for row in predictions]
    genotype_correct = [
        actual_value == predicted_value
        for actual_value, predicted_value in zip(actual_genotypes, predicted_genotypes, strict=True)
    ]
    cohort_correct = [
        actual_value == predicted_value
        for actual_value, predicted_value in zip(actual_cohorts, predicted_cohorts, strict=True)
    ]
    joint_correct = [
        genotype_value and cohort_value
        for genotype_value, cohort_value in zip(genotype_correct, cohort_correct, strict=True)
    ]
    return {
        "joint_accuracy": float(np.mean(joint_correct)),
        "genotype_accuracy": float(np.mean(genotype_correct)),
        "genotype_balanced_accuracy": _balanced_accuracy(actual_genotypes, predicted_genotypes),
        "cohort_accuracy": float(np.mean(cohort_correct)),
        "cohort_balanced_accuracy": _balanced_accuracy(actual_cohorts, predicted_cohorts),
    }


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
    n_side_features: int,
    fusion_hidden_dim: int,
    pooling: str,
    genotype_pooling: str | None,
    cohort_pooling: str | None,
    sequence_unit: str,
    chain_length: int,
    chain_stride: int,
    attention_hidden_dim: int,
    n_genotype_classes: int,
    n_cohort_classes: int,
    dropout: float,
):
    torch, nn = _import_torch()
    attention_pooling_modes = {"attention", "mean_attention_concat"}
    accepted_pooling_modes = {"mean", *attention_pooling_modes}
    accepted_sequence_units = {"segment", "segment_chain"}
    if sequence_unit not in accepted_sequence_units:
        raise ValueError(f"unsupported sequence_unit: {sequence_unit!r}")
    if chain_length < 1:
        raise ValueError(f"chain_length must be at least 1, got {chain_length}")
    if chain_stride < 1:
        raise ValueError(f"chain_stride must be at least 1, got {chain_stride}")
    resolved_genotype_pooling = genotype_pooling or pooling
    resolved_cohort_pooling = cohort_pooling or pooling
    axis_specific_pooling = genotype_pooling is not None or cohort_pooling is not None

    def _validate_pooling(pooling_name: str) -> None:
        if pooling_name not in accepted_pooling_modes:
            raise ValueError(f"unsupported pooling: {pooling_name!r}")

    def _pooled_embedding_dim(pooling_name: str) -> int:
        return embedding_dim * 2 if pooling_name == "mean_attention_concat" else embedding_dim

    _validate_pooling(pooling)
    _validate_pooling(resolved_genotype_pooling)
    _validate_pooling(resolved_cohort_pooling)
    pooled_embedding_dim = _pooled_embedding_dim(pooling)
    genotype_pooled_embedding_dim = _pooled_embedding_dim(resolved_genotype_pooling)
    cohort_pooled_embedding_dim = _pooled_embedding_dim(resolved_cohort_pooling)

    class SegmentConvMeanPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.axis_specific_pooling = axis_specific_pooling
            self.sequence_unit = sequence_unit
            self.chain_length = chain_length
            self.chain_stride = chain_stride
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
            if sequence_unit == "segment_chain":
                self.chain_encoder = nn.Sequential(
                    nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            else:
                self.chain_encoder = None
            self.pooling = pooling
            self.genotype_pooling = resolved_genotype_pooling
            self.cohort_pooling = resolved_cohort_pooling
            if self.axis_specific_pooling:
                self.genotype_attention = self._make_attention(resolved_genotype_pooling)
                self.cohort_attention = self._make_attention(resolved_cohort_pooling)
                self.genotype_fusion = self._make_fusion(genotype_pooled_embedding_dim)
                self.cohort_fusion = self._make_fusion(cohort_pooled_embedding_dim)
                genotype_head_input_dim = fusion_hidden_dim if n_side_features > 0 else genotype_pooled_embedding_dim
                cohort_head_input_dim = fusion_hidden_dim if n_side_features > 0 else cohort_pooled_embedding_dim
            else:
                self.attention = self._make_attention(pooling)
                self.fusion = self._make_fusion(pooled_embedding_dim)
                genotype_head_input_dim = fusion_hidden_dim if n_side_features > 0 else pooled_embedding_dim
                cohort_head_input_dim = genotype_head_input_dim
            self.genotype_head = nn.Linear(genotype_head_input_dim, n_genotype_classes)
            self.cohort_head = nn.Linear(cohort_head_input_dim, n_cohort_classes)

        def _make_attention(self, pooling_name: str):
            if pooling_name not in attention_pooling_modes:
                return None
            return nn.Sequential(
                    nn.Linear(embedding_dim, attention_hidden_dim),
                    nn.Tanh(),
                    nn.Linear(attention_hidden_dim, 1),
                )

        def _make_fusion(self, pooled_dim: int):
            if n_side_features <= 0:
                return None
            return nn.Sequential(
                nn.Linear(pooled_dim + n_side_features, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def encode_segments(self, segments):
            return self.projection(self.segment_encoder(segments))

        def encode_chains(self, segment_embeddings):
            n_segments = segment_embeddings.shape[0]
            if n_segments <= self.chain_length:
                chain_tensor = segment_embeddings.transpose(0, 1).unsqueeze(0)
                return self.chain_encoder(chain_tensor)
            chain_tensors = segment_embeddings.unfold(0, self.chain_length, self.chain_stride)
            return self.chain_encoder(chain_tensors.contiguous())

        def encode_units(self, segments):
            segment_embeddings = self.encode_segments(segments)
            if self.sequence_unit == "segment":
                return segment_embeddings
            return self.encode_chains(segment_embeddings)

        def pool_segments(self, segment_embeddings, pooling_name: str, attention):
            mean_embedding = segment_embeddings.mean(dim=0)
            if pooling_name == "mean":
                return mean_embedding
            attention_logits = attention(segment_embeddings).squeeze(-1)
            attention_weights = torch.softmax(attention_logits, dim=0)
            attention_embedding = (segment_embeddings * attention_weights.unsqueeze(-1)).sum(dim=0)
            if pooling_name == "attention":
                return attention_embedding
            return torch.cat([mean_embedding, attention_embedding], dim=0)

        def _fuse_embedding(self, fly_embedding, side_features, fusion):
            if fusion is None:
                return fly_embedding
            if side_features is None:
                raise ValueError("side_features are required for this fused sequence model")
            return fusion(torch.cat([fly_embedding, side_features], dim=0))

        def forward_fly(self, segments, side_features=None):
            unit_embeddings = self.encode_units(segments)
            if self.axis_specific_pooling:
                genotype_embedding = self.pool_segments(
                    unit_embeddings, self.genotype_pooling, self.genotype_attention
                )
                cohort_embedding = self.pool_segments(unit_embeddings, self.cohort_pooling, self.cohort_attention)
                genotype_embedding = self._fuse_embedding(genotype_embedding, side_features, self.genotype_fusion)
                cohort_embedding = self._fuse_embedding(cohort_embedding, side_features, self.cohort_fusion)
            else:
                fly_embedding = self.pool_segments(unit_embeddings, self.pooling, self.attention)
                fly_embedding = self._fuse_embedding(fly_embedding, side_features, self.fusion)
                genotype_embedding = fly_embedding
                cohort_embedding = fly_embedding
            return self.genotype_head(genotype_embedding), self.cohort_head(cohort_embedding)

    return SegmentConvMeanPool()


def _module_predictions(
    *,
    torch: Any,
    module,
    x_scaled: np.ndarray,
    examples: list[FlySequenceExample],
    side_inputs: dict[str, np.ndarray] | None,
    side_means: np.ndarray,
    side_stds: np.ndarray,
    max_segments_per_fly: int | None,
    sequence_unit: str,
    genotype_labels: list[str],
    cohort_labels: list[str],
    device,
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for example in examples:
            if sequence_unit == "segment_chain":
                indices = _centered_contiguous_indices(example.segment_indices, max_segments_per_fly)
            else:
                indices = _evenly_spaced_indices(example.segment_indices, max_segments_per_fly)
            segment_batch = torch.as_tensor(x_scaled[indices], dtype=torch.float32, device=device).permute(0, 2, 1)
            raw_side_input = _scaled_side_input(example, side_inputs, side_means, side_stds)
            side_tensor = (
                None if raw_side_input is None else torch.as_tensor(raw_side_input, dtype=torch.float32, device=device)
            )
            genotype_logits, cohort_logits = module.forward_fly(segment_batch, side_tensor)
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
    side_inputs: dict[str, np.ndarray] | None = None,
    side_feature_names: list[str] | None = None,
    validation_examples: list[FlySequenceExample] | None = None,
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
    fusion_hidden_dim = int(config.get("fusion_hidden_dim", embedding_dim))
    pooling = str(config.get("pooling", "mean"))
    genotype_pooling = str(config["genotype_pooling"]) if "genotype_pooling" in config else None
    cohort_pooling = str(config["cohort_pooling"]) if "cohort_pooling" in config else None
    sequence_unit = str(config.get("sequence_unit", "segment"))
    chain_length = int(config.get("chain_length", 1))
    chain_stride = int(config.get("chain_stride", 1))
    attention_hidden_dim = int(config.get("attention_hidden_dim", embedding_dim))
    dropout = float(config.get("dropout", 0.1))
    cohort_loss_weight = float(config.get("cohort_loss_weight", 1.0))
    progress_interval = int(config.get("progress_interval", 0))
    progress_label = str(config.get("progress_label", "training"))
    validation_interval = int(config.get("validation_interval", progress_interval))
    validation_monitor_metric = str(config.get("validation_monitor_metric", "joint_accuracy"))
    select_best_epoch = str(config.get("select_best_epoch", "false")).lower() in {"1", "true", "yes"}
    early_stopping_patience = int(config.get("early_stopping_patience", 0))
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
    resolved_side_feature_names = side_feature_names or []
    if side_inputs is None:
        side_means = np.zeros(0, dtype=np.float32)
        side_stds = np.ones(0, dtype=np.float32)
    else:
        side_means, side_stds = _fit_side_scaler(examples, side_inputs)
    device = _resolve_device(torch, config)
    module = _build_module(
        n_channels=x.shape[2],
        conv_channels=conv_channels,
        embedding_dim=embedding_dim,
        n_side_features=len(resolved_side_feature_names),
        fusion_hidden_dim=fusion_hidden_dim,
        pooling=pooling,
        genotype_pooling=genotype_pooling,
        cohort_pooling=cohort_pooling,
        sequence_unit=sequence_unit,
        chain_length=chain_length,
        chain_stride=chain_stride,
        attention_hidden_dim=attention_hidden_dim,
        n_genotype_classes=len(genotype_labels),
        n_cohort_classes=len(cohort_labels),
        dropout=dropout,
    ).to(device)
    pooled_embedding_dim = embedding_dim * 2 if pooling == "mean_attention_concat" else embedding_dim
    genotype_pooled_embedding_dim = (
        embedding_dim * 2 if (genotype_pooling or pooling) == "mean_attention_concat" else embedding_dim
    )
    cohort_pooled_embedding_dim = (
        embedding_dim * 2 if (cohort_pooling or pooling) == "mean_attention_concat" else embedding_dim
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(random_seed + 1)
    best_epoch: int | None = None
    best_validation_metric: float | None = None
    best_state_dict: dict[str, object] | None = None
    epochs_since_best = 0
    validation_history: list[dict[str, object]] = []

    module.train()
    for epoch_idx in range(max_iter):
        optimizer.zero_grad()
        total_loss = 0.0
        for example in examples:
            if sequence_unit == "segment_chain":
                indices = _random_contiguous_indices(example.segment_indices, train_max_segments_per_fly, rng)
            else:
                indices = _random_sample_indices(example.segment_indices, train_max_segments_per_fly, rng)
            segment_batch = torch.as_tensor(x_scaled[indices], dtype=torch.float32, device=device).permute(0, 2, 1)
            raw_side_input = _scaled_side_input(example, side_inputs, side_means, side_stds)
            side_tensor = (
                None if raw_side_input is None else torch.as_tensor(raw_side_input, dtype=torch.float32, device=device)
            )
            genotype_logits, cohort_logits = module.forward_fly(segment_batch, side_tensor)
            genotype_target = torch.tensor([genotype_to_index[example.genotype]], dtype=torch.long, device=device)
            cohort_target = torch.tensor([cohort_to_index[example.cohort]], dtype=torch.long, device=device)
            loss = criterion(genotype_logits.unsqueeze(0), genotype_target)
            loss = loss + cohort_loss_weight * criterion(cohort_logits.unsqueeze(0), cohort_target)
            total_loss += float(loss.detach().cpu())
            (loss / len(examples)).backward()
        optimizer.step()
        mean_loss = total_loss / len(examples)
        do_progress = progress_interval > 0 and (
            epoch_idx == 0 or (epoch_idx + 1) % progress_interval == 0 or epoch_idx + 1 == max_iter
        )
        do_validation = validation_examples and validation_interval > 0 and (
            epoch_idx == 0 or (epoch_idx + 1) % validation_interval == 0 or epoch_idx + 1 == max_iter
        )
        validation_metrics = None
        metric_value = None
        if do_validation:
            module.eval()
            validation_predictions = _module_predictions(
                torch=torch,
                module=module,
                x_scaled=x_scaled,
                examples=validation_examples,
                side_inputs=side_inputs,
                side_means=side_means,
                side_stds=side_stds,
                max_segments_per_fly=eval_max_segments_per_fly,
                sequence_unit=sequence_unit,
                genotype_labels=genotype_labels,
                cohort_labels=cohort_labels,
                device=device,
            )
            module.train()
            validation_metrics = _prediction_metrics(validation_predictions)
            if validation_monitor_metric not in validation_metrics:
                raise ValueError(
                    f"unsupported validation_monitor_metric: {validation_monitor_metric!r}; "
                    f"expected one of {sorted(validation_metrics)}"
                )
            metric_value = validation_metrics[validation_monitor_metric]
            validation_history.append(
                {
                    "epoch": epoch_idx + 1,
                    "train_loss": mean_loss,
                    **validation_metrics,
                }
            )
            if best_validation_metric is None or metric_value > best_validation_metric:
                best_validation_metric = metric_value
                best_epoch = epoch_idx + 1
                epochs_since_best = 0
                if select_best_epoch:
                    best_state_dict = {
                        key: value.detach().clone()
                        for key, value in module.state_dict().items()
                    }
            else:
                epochs_since_best += 1
        if do_progress or do_validation:
            progress_message = f"{progress_label}: epoch {epoch_idx + 1}/{max_iter} loss={mean_loss:.4f}"
            if validation_metrics is not None and metric_value is not None:
                progress_message += (
                    f" valid_joint={validation_metrics['joint_accuracy']:.3f}"
                    f" valid_genotype={validation_metrics['genotype_accuracy']:.3f}"
                    f" valid_cohort={validation_metrics['cohort_accuracy']:.3f}"
                    f" monitor_{validation_monitor_metric}={metric_value:.3f}"
                )
            print(progress_message, flush=True)
        if (
            do_validation
            and select_best_epoch
            and early_stopping_patience > 0
            and epochs_since_best >= early_stopping_patience
        ):
            print(
                f"{progress_label}: stopping early at epoch {epoch_idx + 1}; "
                f"best_epoch={best_epoch} "
                f"best_{validation_monitor_metric}={best_validation_metric:.3f}",
                flush=True,
            )
            break

    if select_best_epoch and best_state_dict is not None:
        module.load_state_dict(best_state_dict)

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
        "pooled_embedding_dim": pooled_embedding_dim,
        "genotype_pooling": genotype_pooling,
        "cohort_pooling": cohort_pooling,
        "genotype_pooled_embedding_dim": genotype_pooled_embedding_dim,
        "cohort_pooled_embedding_dim": cohort_pooled_embedding_dim,
        "fusion_hidden_dim": fusion_hidden_dim,
        "pooling": pooling,
        "sequence_unit": sequence_unit,
        "chain_length": chain_length,
        "chain_stride": chain_stride,
        "attention_hidden_dim": attention_hidden_dim,
        "dropout": dropout,
        "side_feature_names": resolved_side_feature_names,
        "side_input_means": side_means,
        "side_input_stds": side_stds,
        "n_side_features": len(resolved_side_feature_names),
        "cohort_loss_weight": cohort_loss_weight,
        "progress_interval": progress_interval,
        "validation_interval": validation_interval,
        "validation_monitor_metric": validation_monitor_metric,
        "select_best_epoch": select_best_epoch,
        "best_epoch": best_epoch,
        "best_validation_metric": best_validation_metric,
        "validation_history": validation_history,
        "early_stopping_patience": early_stopping_patience,
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": str(device),
        "random_seed": random_seed,
        "train_max_segments_per_fly": 0 if train_max_segments_per_fly is None else train_max_segments_per_fly,
        "eval_max_segments_per_fly": 0 if eval_max_segments_per_fly is None else eval_max_segments_per_fly,
        "scaler_max_segments_per_fly": 0 if scaler_max_segments_per_fly is None else scaler_max_segments_per_fly,
        "segment_sampling": (
            "random_contiguous_span_per_epoch"
            if sequence_unit == "segment_chain"
            else "random_without_replacement_per_epoch"
        ),
    }


def predict_torch_sequence_meanpool(
    x: np.ndarray,
    examples: list[FlySequenceExample],
    *,
    model: dict[str, object],
    side_inputs: dict[str, np.ndarray] | None = None,
) -> list[dict[str, object]]:
    torch, _ = _import_torch()
    module = model["module"]
    device = torch.device(str(model.get("device", "cpu")))
    module.eval()
    means = np.asarray(model["input_means"], dtype=np.float32)
    stds = np.asarray(model["input_stds"], dtype=np.float32)
    x_scaled = _scaled_sequences(x, means, stds)
    side_feature_names = [str(name) for name in model.get("side_feature_names", [])]
    if side_feature_names:
        if side_inputs is None:
            raise ValueError("side_inputs are required to predict with this fused sequence model")
        side_means = np.asarray(model["side_input_means"], dtype=np.float32)
        side_stds = np.asarray(model["side_input_stds"], dtype=np.float32)
    else:
        side_means = np.zeros(0, dtype=np.float32)
        side_stds = np.ones(0, dtype=np.float32)
    max_segments_per_fly = int(model.get("eval_max_segments_per_fly", 0)) or None
    sequence_unit = str(model.get("sequence_unit", "segment"))
    genotype_labels = [str(label) for label in model["genotype_labels"]]
    cohort_labels = [str(label) for label in model["cohort_labels"]]
    return _module_predictions(
        torch=torch,
        module=module,
        x_scaled=x_scaled,
        examples=examples,
        side_inputs=side_inputs,
        side_means=side_means,
        side_stds=side_stds,
        max_segments_per_fly=max_segments_per_fly,
        sequence_unit=sequence_unit,
        genotype_labels=genotype_labels,
        cohort_labels=cohort_labels,
        device=device,
    )


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
