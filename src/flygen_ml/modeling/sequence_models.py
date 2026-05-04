from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlySequenceExample:
    fly_id: str
    sample_key: str
    genotype: str
    cohort: str
    segment_indices: np.ndarray
    n_segments: int
    n_segments_with_qc_flags: int


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(np.clip(shifted, -30.0, 30.0))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def _cross_entropy_grad(probs: np.ndarray, target: int) -> np.ndarray:
    grad = probs.copy()
    grad[target] -= 1.0
    return grad


def _evenly_spaced_indices(indices: np.ndarray, max_count: int | None) -> np.ndarray:
    if max_count is None or max_count <= 0 or len(indices) <= max_count:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=max_count).round().astype(int)
    return indices[positions]


def load_sequence_npz(path: str) -> tuple[np.ndarray, list[FlySequenceExample], dict[str, object]]:
    payload = np.load(path)
    x = np.asarray(payload["x"], dtype=np.float32)
    fly_ids = [str(value) for value in payload["fly_id"]]
    sample_keys = [str(value) for value in payload["sample_key"]]
    genotypes = [str(value) for value in payload["genotype"]]
    cohorts = [str(value) for value in payload["cohort"]]
    qc_flags = [str(value) for value in payload["qc_flags"]] if "qc_flags" in payload else [""] * len(fly_ids)

    grouped: dict[str, list[int]] = {}
    for idx, fly_id in enumerate(fly_ids):
        grouped.setdefault(fly_id, []).append(idx)

    examples: list[FlySequenceExample] = []
    for fly_id, indices_raw in sorted(grouped.items()):
        indices = np.asarray(indices_raw, dtype=int)
        first = indices[0]
        genotype_values = {genotypes[idx] for idx in indices}
        cohort_values = {cohorts[idx] for idx in indices}
        if len(genotype_values) != 1:
            raise ValueError(f"fly {fly_id!r} has inconsistent genotype labels: {sorted(genotype_values)}")
        if len(cohort_values) != 1:
            raise ValueError(f"fly {fly_id!r} has inconsistent cohort labels: {sorted(cohort_values)}")
        cohort = cohorts[first]
        if cohort == "":
            raise ValueError("sequence training requires non-empty cohort labels")
        examples.append(
            FlySequenceExample(
                fly_id=fly_id,
                sample_key=sample_keys[first],
                genotype=genotypes[first],
                cohort=cohort,
                segment_indices=indices,
                n_segments=len(indices),
                n_segments_with_qc_flags=sum(bool(qc_flags[idx]) for idx in indices),
            )
        )
    metadata = {
        "channels": [str(value) for value in payload["channels"]],
        "target_length": int(payload["target_length"]),
        "n_segments": int(x.shape[0]),
        "n_flies": len(examples),
    }
    return x, examples, metadata


def _fit_scaler(
    x: np.ndarray,
    examples: list[FlySequenceExample],
    max_segments_per_fly: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.concatenate(
        [_evenly_spaced_indices(example.segment_indices, max_segments_per_fly) for example in examples]
    )
    flat = x[selected].reshape(len(selected), -1)
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    stds = np.where(stds > 1e-6, stds, 1.0)
    return means.astype(np.float32), stds.astype(np.float32)


def _init_model(
    *,
    input_dim: int,
    hidden_dim: int,
    genotype_labels: list[str],
    cohort_labels: list[str],
    random_seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(random_seed)
    scale = 1.0 / max(1, input_dim) ** 0.5
    return {
        "model_kind": "sequence_meanpool_mlp_numpy_v1",
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "genotype_labels": genotype_labels,
        "cohort_labels": cohort_labels,
        "encoder_weight": rng.normal(0.0, scale, size=(input_dim, hidden_dim)).astype(np.float32),
        "encoder_bias": np.zeros(hidden_dim, dtype=np.float32),
        "genotype_weight": rng.normal(
            0.0,
            1.0 / hidden_dim**0.5,
            size=(hidden_dim, len(genotype_labels)),
        ).astype(np.float32),
        "genotype_bias": np.zeros(len(genotype_labels), dtype=np.float32),
        "cohort_weight": rng.normal(
            0.0,
            1.0 / hidden_dim**0.5,
            size=(hidden_dim, len(cohort_labels)),
        ).astype(np.float32),
        "cohort_bias": np.zeros(len(cohort_labels), dtype=np.float32),
    }


def _predict_one(
    flat_scaled: np.ndarray,
    example: FlySequenceExample,
    model: dict[str, object],
    *,
    max_segments_per_fly: int | None,
) -> dict[str, object]:
    indices = _evenly_spaced_indices(example.segment_indices, max_segments_per_fly)
    w1 = np.asarray(model["encoder_weight"], dtype=np.float32)
    b1 = np.asarray(model["encoder_bias"], dtype=np.float32)
    wg = np.asarray(model["genotype_weight"], dtype=np.float32)
    bg = np.asarray(model["genotype_bias"], dtype=np.float32)
    wc = np.asarray(model["cohort_weight"], dtype=np.float32)
    bc = np.asarray(model["cohort_bias"], dtype=np.float32)
    segment_embeddings = np.tanh(flat_scaled[indices] @ w1 + b1)
    fly_embedding = segment_embeddings.mean(axis=0)
    genotype_probs = _softmax(fly_embedding @ wg + bg)
    cohort_probs = _softmax(fly_embedding @ wc + bc)
    genotype_idx = int(genotype_probs.argmax())
    cohort_idx = int(cohort_probs.argmax())
    genotype_labels = [str(label) for label in model["genotype_labels"]]
    cohort_labels = [str(label) for label in model["cohort_labels"]]
    return {
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


def predict_sequence_meanpool(
    x: np.ndarray,
    examples: list[FlySequenceExample],
    *,
    model: dict[str, object],
) -> list[dict[str, object]]:
    means = np.asarray(model["input_means"], dtype=np.float32)
    stds = np.asarray(model["input_stds"], dtype=np.float32)
    flat_scaled = (x.reshape(x.shape[0], -1) - means) / stds
    max_segments_per_fly = int(model.get("max_segments_per_fly", 0)) or None
    return [
        _predict_one(flat_scaled, example, model, max_segments_per_fly=max_segments_per_fly)
        for example in examples
    ]


def train_sequence_meanpool(
    x: np.ndarray,
    examples: list[FlySequenceExample],
    *,
    config: dict[str, object],
) -> dict[str, object]:
    if not examples:
        raise ValueError("cannot train sequence model with no fly examples")
    hidden_dim = int(config.get("hidden_dim", 32))
    max_iter = int(config.get("max_iter", 100))
    learning_rate = float(config.get("learning_rate", 0.01))
    l2_reg = float(config.get("l2_reg", 0.0001))
    random_seed = int(config.get("random_seed", 0))
    max_segments_per_fly = int(config.get("max_segments_per_fly", 200))

    genotype_labels = sorted({example.genotype for example in examples})
    cohort_labels = sorted({example.cohort for example in examples})
    if len(genotype_labels) < 2:
        raise ValueError(f"expected at least two genotype labels, got {genotype_labels}")
    if len(cohort_labels) < 2:
        raise ValueError(f"expected at least two cohort labels, got {cohort_labels}")
    genotype_to_index = {label: idx for idx, label in enumerate(genotype_labels)}
    cohort_to_index = {label: idx for idx, label in enumerate(cohort_labels)}

    means, stds = _fit_scaler(x, examples, max_segments_per_fly)
    flat_scaled = (x.reshape(x.shape[0], -1) - means) / stds
    model = _init_model(
        input_dim=flat_scaled.shape[1],
        hidden_dim=hidden_dim,
        genotype_labels=genotype_labels,
        cohort_labels=cohort_labels,
        random_seed=random_seed,
    )
    model["input_means"] = means
    model["input_stds"] = stds
    model["max_segments_per_fly"] = max_segments_per_fly

    w1 = np.asarray(model["encoder_weight"], dtype=np.float32)
    b1 = np.asarray(model["encoder_bias"], dtype=np.float32)
    wg = np.asarray(model["genotype_weight"], dtype=np.float32)
    bg = np.asarray(model["genotype_bias"], dtype=np.float32)
    wc = np.asarray(model["cohort_weight"], dtype=np.float32)
    bc = np.asarray(model["cohort_bias"], dtype=np.float32)

    for _ in range(max_iter):
        grad_w1 = np.zeros_like(w1)
        grad_b1 = np.zeros_like(b1)
        grad_wg = np.zeros_like(wg)
        grad_bg = np.zeros_like(bg)
        grad_wc = np.zeros_like(wc)
        grad_bc = np.zeros_like(bc)
        for example in examples:
            indices = _evenly_spaced_indices(example.segment_indices, max_segments_per_fly)
            segment_inputs = flat_scaled[indices]
            hidden_pre = segment_inputs @ w1 + b1
            segment_embeddings = np.tanh(hidden_pre)
            fly_embedding = segment_embeddings.mean(axis=0)

            genotype_probs = _softmax(fly_embedding @ wg + bg)
            cohort_probs = _softmax(fly_embedding @ wc + bc)
            grad_genotype_logits = _cross_entropy_grad(genotype_probs, genotype_to_index[example.genotype])
            grad_cohort_logits = _cross_entropy_grad(cohort_probs, cohort_to_index[example.cohort])

            grad_wg += np.outer(fly_embedding, grad_genotype_logits)
            grad_bg += grad_genotype_logits
            grad_wc += np.outer(fly_embedding, grad_cohort_logits)
            grad_bc += grad_cohort_logits
            grad_fly_embedding = grad_genotype_logits @ wg.T + grad_cohort_logits @ wc.T
            grad_segment_embedding = grad_fly_embedding / len(indices)
            grad_hidden_pre = (1.0 - segment_embeddings**2) * grad_segment_embedding
            grad_w1 += segment_inputs.T @ grad_hidden_pre
            grad_b1 += grad_hidden_pre.sum(axis=0)

        scale = 1.0 / len(examples)
        w1 -= learning_rate * (grad_w1 * scale + l2_reg * w1)
        b1 -= learning_rate * grad_b1 * scale
        wg -= learning_rate * (grad_wg * scale + l2_reg * wg)
        bg -= learning_rate * grad_bg * scale
        wc -= learning_rate * (grad_wc * scale + l2_reg * wc)
        bc -= learning_rate * grad_bc * scale

    model.update(
        {
            "encoder_weight": w1,
            "encoder_bias": b1,
            "genotype_weight": wg,
            "genotype_bias": bg,
            "cohort_weight": wc,
            "cohort_bias": bc,
        }
    )
    return model


def serializable_sequence_model(model: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in model.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        else:
            payload[key] = value
    return payload
