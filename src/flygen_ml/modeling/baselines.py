from __future__ import annotations

import numpy as np


NON_FEATURE_COLUMNS = {
    "fly_id",
    "sample_key",
    "genotype",
    "chamber_type",
    "training_idx",
}


def _resolve_feature_names(
    rows: list[dict[str, object]],
    *,
    exclude_feature_names: set[str] | None = None,
) -> list[str]:
    first = rows[0]
    excluded = exclude_feature_names or set()
    feature_names = [
        key
        for key, value in first.items()
        if key not in NON_FEATURE_COLUMNS and key not in excluded and isinstance(value, (int, float))
    ]
    if not feature_names:
        raise ValueError("no numeric feature columns found")
    return sorted(feature_names)


def _matrix_from_rows(rows: list[dict[str, object]], feature_names: list[str]) -> np.ndarray:
    matrix = np.empty((len(rows), len(feature_names)), dtype=float)
    for row_idx, row in enumerate(rows):
        for col_idx, feature_name in enumerate(feature_names):
            value = row.get(feature_name)
            matrix[row_idx, col_idx] = float(value) if isinstance(value, (int, float)) else float("nan")
    return matrix


def _column_means(x: np.ndarray) -> np.ndarray:
    means = np.empty(x.shape[1], dtype=float)
    for idx in range(x.shape[1]):
        col = x[:, idx]
        finite = col[np.isfinite(col)]
        means[idx] = float(finite.mean()) if finite.size else 0.0
    return means


def _column_stds(x: np.ndarray) -> np.ndarray:
    stds = np.empty(x.shape[1], dtype=float)
    for idx in range(x.shape[1]):
        col = x[:, idx]
        finite = col[np.isfinite(col)]
        if finite.size < 2:
            stds[idx] = 1.0
        else:
            std_value = float(finite.std())
            stds[idx] = std_value if std_value > 0 else 1.0
    return stds


def _impute_and_scale(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    imputed = np.where(np.isfinite(x), x, means)
    return (imputed - means) / stds


def _binary_targets(rows: list[dict[str, object]]) -> tuple[np.ndarray, list[str]]:
    labels = sorted({str(row["genotype"]) for row in rows})
    if len(labels) != 2:
        raise ValueError(f"baseline expects exactly 2 genotypes, got {labels}")
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    y = np.asarray([label_to_index[str(row["genotype"])] for row in rows], dtype=float)
    return y, labels


def _class_targets(rows: list[dict[str, object]]) -> tuple[np.ndarray, list[str]]:
    labels = sorted({str(row["genotype"]) for row in rows})
    if len(labels) < 2:
        raise ValueError(f"baseline expects at least 2 genotypes, got {labels}")
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    y = np.asarray([label_to_index[str(row["genotype"])] for row in rows], dtype=int)
    return y, labels


def _fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    learning_rate: float,
    max_iter: int,
    l2_reg: float,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(x_train.shape[1], dtype=float)
    bias = 0.0
    n_samples = float(len(y_train))
    for _ in range(max_iter):
        logits = x_train @ weights + bias
        logits = np.clip(logits, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - y_train
        grad_w = (x_train.T @ error) / n_samples + l2_reg * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, bias


def _predict_probabilities(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    logits = np.clip(x @ weights + bias, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _fit_softmax_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_classes: int,
    learning_rate: float,
    max_iter: int,
    l2_reg: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros((x_train.shape[1], n_classes), dtype=float)
    bias = np.zeros(n_classes, dtype=float)
    y_one_hot = np.eye(n_classes, dtype=float)[y_train]
    n_samples = float(len(y_train))
    for _ in range(max_iter):
        logits = x_train @ weights + bias
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(np.clip(logits, -30.0, 30.0))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        error = probs - y_one_hot
        grad_w = (x_train.T @ error) / n_samples + l2_reg * weights
        grad_b = error.mean(axis=0)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, bias


def _predict_class_probabilities(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    logits = x @ weights + bias
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits, -30.0, 30.0))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def train_fly_level_baseline(
    rows: list[dict[str, object]],
    *,
    config: dict[str, object],
) -> dict[str, object]:
    exclude_feature_names = {
        str(name).strip()
        for name in str(config.get("exclude_feature_names", "")).split(",")
        if str(name).strip()
    }
    feature_names = _resolve_feature_names(rows, exclude_feature_names=exclude_feature_names)
    x_train_raw = _matrix_from_rows(rows, feature_names)
    y_train, labels = _class_targets(rows)

    means = _column_means(x_train_raw)
    stds = _column_stds(x_train_raw)
    x_train = _impute_and_scale(x_train_raw, means, stds)

    learning_rate = float(config.get("learning_rate", 0.1))
    max_iter = int(config.get("max_iter", 400))
    l2_reg = float(config.get("l2_reg", 0.01))
    if len(labels) > 2:
        weights, bias = _fit_softmax_regression(
            x_train,
            y_train,
            n_classes=len(labels),
            learning_rate=learning_rate,
            max_iter=max_iter,
            l2_reg=l2_reg,
        )
        train_probs = _predict_class_probabilities(x_train, weights, bias)
        return {
            "model_kind": "softmax_logreg_numpy_v1",
            "feature_names": feature_names,
            "excluded_feature_names": sorted(exclude_feature_names),
            "feature_means": means.tolist(),
            "feature_stds": stds.tolist(),
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "labels": labels,
            "train_probabilities": train_probs.tolist(),
        }

    y_train_binary = y_train.astype(float)
    weights, bias = _fit_logistic_regression(
        x_train,
        y_train_binary,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_reg=l2_reg,
    )
    train_probs = _predict_probabilities(x_train, weights, bias)
    return {
        "model_kind": "logreg_numpy_v1",
        "feature_names": feature_names,
        "excluded_feature_names": sorted(exclude_feature_names),
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "labels": labels,
        "train_probabilities": train_probs.tolist(),
    }


def predict_fly_level_baseline(
    rows: list[dict[str, object]],
    *,
    model: dict[str, object],
) -> list[dict[str, object]]:
    feature_names = [str(name) for name in model["feature_names"]]
    x_raw = _matrix_from_rows(rows, feature_names)
    means = np.asarray(model["feature_means"], dtype=float)
    stds = np.asarray(model["feature_stds"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    model_kind = str(model.get("model_kind", "logreg_numpy_v1"))
    labels = [str(label) for label in model["labels"]]

    x = _impute_and_scale(x_raw, means, stds)
    if model_kind == "softmax_logreg_numpy_v1":
        bias = np.asarray(model["bias"], dtype=float)
        class_probs = _predict_class_probabilities(x, weights, bias)
        predictions: list[dict[str, object]] = []
        for row, probs in zip(rows, class_probs):
            predicted_idx = int(probs.argmax())
            predictions.append(
                {
                    "fly_id": row["fly_id"],
                    "sample_key": row["sample_key"],
                    "actual_genotype": row["genotype"],
                    "predicted_genotype": labels[predicted_idx],
                    "predicted_probability": float(probs[predicted_idx]),
                    "n_segments": row.get("n_segments", ""),
                    "n_segments_with_qc_flags": row.get("n_segments_with_qc_flags", ""),
                }
            )
        return predictions

    bias = float(model["bias"])
    positive_probs = _predict_probabilities(x, weights, bias)
    predictions: list[dict[str, object]] = []
    for row, positive_prob in zip(rows, positive_probs):
        prob = float(positive_prob)
        predicted_label = labels[1] if prob >= 0.5 else labels[0]
        predictions.append(
            {
                "fly_id": row["fly_id"],
                "sample_key": row["sample_key"],
                "actual_genotype": row["genotype"],
                "predicted_genotype": predicted_label,
                "predicted_probability": prob,
                "n_segments": row.get("n_segments", ""),
                "n_segments_with_qc_flags": row.get("n_segments_with_qc_flags", ""),
            }
        )
    return predictions
