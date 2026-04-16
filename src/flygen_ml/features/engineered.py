from __future__ import annotations

from flygen_ml.schema import SegmentRecord


def compute_engineered_features(segment: SegmentRecord) -> dict[str, float]:
    del segment
    # TODO: add the first interpretable feature set once canonical segment extraction is verified.
    raise NotImplementedError("Engineered feature extraction is not implemented yet.")
