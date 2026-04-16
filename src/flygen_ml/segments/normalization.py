from __future__ import annotations

from flygen_ml.schema import SegmentRecord


def normalize_segment_coordinates(segment: SegmentRecord):
    del segment
    # TODO: implement reward-centered coordinate normalization once segment semantics are locked.
    raise NotImplementedError("Segment normalization is not implemented yet.")
