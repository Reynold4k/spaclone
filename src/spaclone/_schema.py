"""Column names and validation of the receptor ("air") table.

Every public function takes ``*_key`` keyword arguments that default to the constants below, so tables with other
column names can be used without renaming. Coordinates are always micrometres.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

SPOT = "spot_id"
X = "x_um"
Y = "y_um"
LOCUS = "locus"
CLONE = "clone_id"
JUNCTION_AA = "junction_aa"
JUNCTION = "junction"
V_CALL = "v_call"
J_CALL = "j_call"
C_CALL = "c_call"
PRODUCTIVE = "productive"
READS = "consensus_count"
UMI = "umi"
LIBRARY = "library_id"
LOCUS_ID = "locus_id"

REQUIRED = (SPOT, X, Y, LOCUS, CLONE)

#: Side of one Stereo-seq DNB (bin1) in micrometres.
DNB_UM = 0.5
#: Curio Seeker / AIR-SPACE bead coordinate unit in micrometres.
CURIO_UM = 0.65

HEAVY_LOCI = ("IGH", "TRB", "TRD")
LIGHT_LOCI = ("IGK", "IGL", "TRA", "TRG")
BCR_LOCI = ("IGH", "IGK", "IGL")
TCR_LOCI = ("TRA", "TRB", "TRG", "TRD")
ALL_LOCI = BCR_LOCI + TCR_LOCI


def validate_air(air: pd.DataFrame, *, required: Iterable[str] = REQUIRED, numeric: Iterable[str] = (X, Y)) -> None:
    """Check that a receptor table has the columns and types the package relies on.

    Parameters
    ----------
    air
        Receptor table, one row per molecule, contig or cell call.
    required
        Columns that must be present.
    numeric
        Columns that must be numeric and finite.

    Raises
    ------
    ValueError
        Naming the first missing or malformed column.
    """
    if not isinstance(air, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(air).__name__}")
    missing = [c for c in required if c not in air.columns]
    if missing:
        raise ValueError(
            f"receptor table lacks column(s) {missing}; pass the matching *_key arguments if your columns are named "
            f"differently (expected names: {list(required)})"
        )
    for c in numeric:
        if c not in air.columns:
            continue
        if not pd.api.types.is_numeric_dtype(air[c]):
            raise ValueError(f"column {c!r} must be numeric (micrometres), got dtype {air[c].dtype}")
        if len(air) and not np.isfinite(air[c].to_numpy(dtype=float)).all():
            raise ValueError(f"column {c!r} contains NaN or infinite values")


def coords(df: pd.DataFrame, *, x_key: str = X, y_key: str = Y) -> np.ndarray:
    """Return an ``(n, 2)`` float array of coordinates in micrometres."""
    return np.column_stack([df[x_key].to_numpy(dtype=float), df[y_key].to_numpy(dtype=float)])
