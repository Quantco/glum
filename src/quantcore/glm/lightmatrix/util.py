from typing import Optional, Tuple

import numpy as np


def _set_up_rows_or_cols(
    arr: Optional[np.ndarray], length: int, dtype=np.int32
) -> np.ndarray:
    if arr is None:
        return np.arange(length, dtype=dtype)
    return np.asarray(arr).astype(dtype)


def _setup_restrictions(
    shape: Tuple[int, int],
    rows: Optional[np.ndarray],
    cols: Optional[np.ndarray],
    dtype=np.int32,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = _set_up_rows_or_cols(rows, shape[0], dtype)
    cols = _set_up_rows_or_cols(cols, shape[1], dtype)
    return rows, cols


def _check_out_shape(out: Optional[np.ndarray], expected_first_dim: int) -> None:
    if out is not None and out.shape[0] != expected_first_dim:
        raise ValueError(
            f"""The first dimension of 'out' must be {expected_first_dim}, but it is
            {out.shape[0]}."""
        )


def _check_transpose_matvec_out_shape(mat, out: Optional[np.ndarray]) -> None:
    _check_out_shape(out, mat.shape[1])


def _check_matvec_out_shape(mat, out: Optional[np.ndarray]) -> None:
    _check_out_shape(out, mat.shape[0])
