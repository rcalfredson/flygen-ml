from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np


class _LegacyPlaceholder:
    """Placeholder used to tolerate selected legacy pickled references."""


class LegacyCompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: D401
        # TODO: fixture-verify whether more legacy classes need explicit handling.
        if module == "__main__":
            return _LegacyPlaceholder
        return super().find_class(module, name)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        with warnings.catch_warnings():
            # Some legacy NumPy dtypes inside upstream-style pickles emit a
            # compatibility warning under newer NumPy releases during unpickling.
            warnings.filterwarnings(
                "ignore",
                message=r".*align should be passed as Python or NumPy boolean.*",
                category=np.exceptions.VisibleDeprecationWarning,
            )
            return LegacyCompatibleUnpickler(handle, encoding="latin1").load()


def load_recording_pair(data_path: str | Path, trx_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_data = load_pickle(data_path)
    raw_trx = load_pickle(trx_path)
    if not isinstance(raw_data, dict):
        raise TypeError(f"expected .data pickle to load as dict, got {type(raw_data)!r}")
    if not isinstance(raw_trx, dict):
        raise TypeError(f"expected .trx pickle to load as dict, got {type(raw_trx)!r}")
    return raw_data, raw_trx
