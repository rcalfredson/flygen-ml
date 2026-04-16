from __future__ import annotations

from flygen_ml.schema import SegmentRecord, SequenceSample


def build_sequence_sample(segment: SegmentRecord) -> SequenceSample:
    del segment
    # TODO: implement fixed-length sequence building after normalization details are verified.
    raise NotImplementedError("Sequence sample building is not implemented yet.")
