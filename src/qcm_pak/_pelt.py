"""
PELT changepoint search with an L2 (mean-shift) cost.

A vectorised stand-in for ``ruptures.Pelt(model="l2", jump=5)``. It returns the
same breakpoints but takes seconds instead of minutes on a long trace: the cost
of any segment is read from running sums instead of recomputed sample by sample.

The search follows the ``ruptures`` recursion step for step, including which
points it keeps as candidates, so results match the library on the same input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pelt_l2(
    signal: NDArray[np.float64], pen: float, min_size: int, jump: int = 5
) -> list[int]:
    """Find mean-shift breakpoints by pruned exact search.

    Parameters
    ----------
    signal:
        One-dimensional signal.
    pen:
        Cost added for every segment; larger values give fewer breakpoints.
    min_size:
        Shortest allowed segment, in samples.
    jump:
        Breakpoints are only considered every ``jump`` samples.

    Returns
    -------
    list[int]
        Sorted segment ends. Each is the first index of the next segment, and
        the last is ``len(signal)``.
    """
    n = len(signal)
    if n < max(min_size, 1):
        return [n]
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    sum1 = np.concatenate(([0.0], np.cumsum(x)))
    sum2 = np.concatenate(([0.0], np.cumsum(x * x)))

    best = np.full(n + 1, np.nan)    # best total cost of signal[:t]
    best[0] = 0.0
    previous = np.zeros(n + 1, dtype=np.intp)

    ends = [k for k in range(0, n, jump) if k >= min_size]
    ends.append(n)
    admissible: list[int] = []
    for end in ends:
        admissible.append((end - min_size) // jump * jump)
        starts = np.array(
            [t for t in admissible if not np.isnan(best[t])], dtype=np.intp
        )
        length = end - starts
        cost = (sum2[end] - sum2[starts]) - (sum1[end] - sum1[starts]) ** 2 / length
        total = best[starts] + (cost + pen)
        j = int(np.argmin(total))
        best[end] = total[j]
        previous[end] = starts[j]
        # Keep a start only while it can still lead to the best partition.
        # Zipping against ``starts`` (not ``admissible``) is deliberate: it is
        # what ruptures does, and it keeps the two implementations identical.
        keep = total <= total[j] + pen
        admissible = [t for t, k in zip(admissible, keep, strict=False) if k]

    breakpoints = [n]
    while breakpoints[-1] > 0:
        breakpoints.append(int(previous[breakpoints[-1]]))
    return sorted(b for b in breakpoints if b > 0)
