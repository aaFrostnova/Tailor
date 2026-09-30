"""The one place that decides which images a measurement runs on.

The surrogate was measured on a single source (pool E, high-resolution photographs) while the reported
matrix is measured across all five. A curve fitted on one population was therefore used to make
decisions that were then scored on another, with the fitting population making up a tenth of the
evaluation set. Measured, that mismatch is usually small -- the between-source standard deviation is
0.005 at the median -- but its 90th percentile is 0.023 and its maximum 0.042, which is the same size as
the margin the solver leaves itself when it sits a strength exactly on a constraint. Most of the time it
does not matter; on a boundary solution it flips the verdict.

Sampling here follows the evaluation set's own composition, so the table and the matrix describe the
same population. Selection is deterministic given (n, offset): a campaign and its held-out validation
can ask for disjoint slices and know they do not overlap.
"""
import glob, os

POOL = "/data/tailor/workspace/wm_dataset10k/pool"
# the proportions the reported evaluation subset uses: A/B/C real+generated, D DALL-E 3, E high-res real
COMPOSITION = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.15, "E": 0.10}

__all__ = ["sample", "composition_of"]


def _files(src):
    return sorted(glob.glob(os.path.join(POOL, src, "img", "*.png")))


def sample(n, offset=0, composition=None):
    """`n` image paths drawn across the sources in the evaluation set's proportions.

    `offset` skips that many images WITHIN EACH SOURCE, so sample(100) and sample(40, offset=100) are
    disjoint by construction -- the property a held-out validation needs.
    """
    comp = composition or COMPOSITION
    out = []
    for src, frac in sorted(comp.items()):
        k = int(round(n * frac))
        fs = _files(src)
        if not fs:
            continue
        take = fs[offset:offset + k]
        if len(take) < k:                       # a small source wraps rather than silently under-filling
            take = (fs[offset:] + fs)[:k]
        out += take
    # top up from the largest source if rounding left the count short
    if len(out) < n:
        big = max(comp, key=lambda s: len(_files(s)))
        extra = [f for f in _files(big)[offset:] if f not in out]
        out += extra[:n - len(out)]
    return out[:n]


def composition_of(paths):
    """Which sources a set of paths came from -- for stamping provenance on a measurement."""
    c = {}
    for p in paths:
        parts = p.split(f"{POOL}/")
        s = parts[1].split("/")[0] if len(parts) > 1 else "?"
        c[s] = c.get(s, 0) + 1
    return dict(sorted(c.items()))
