"""Rotation candidate generation that depends on no payload fragment.

The geometric cascade used to drive its blind angle sweep with one of the payload fragments, picked by
a hardcoded preference, and to abort on a bit-accuracy gate computed from that same fragment. That is
circular: it asks a mark that the transform may already have destroyed how to undo the transform, and a
configuration whose selected fragments happen to be rotation-fragile makes the sweep inert. Which
fragments a request selects then decides whether rotation is recoverable at all -- a property of the
search leaking into a property of the threat model.

This module proposes angles from the image alone. Two mechanisms, in order of confidence:

  corner geometry -- an in-place rotation leaves wedges of fill colour at the frame corners, and the
      boundary between fill and content is a straight line whose slope IS the rotation. Where those
      wedges exist this is essentially exact (median error 0.00 degrees over the pool).
  exhaustive grid -- otherwise, a fixed sweep over the search range. Uninformed, but complete, and
      independent of every fragment.

A spectral estimator was tried first and is not used: photographs carry a strong per-image orientation
of their own, so aligning the Fourier magnitude with the axes recovers that bias rather than the
rotation. On unrotated images it placed the peak away from zero on three of four, which is
disqualifying for an absolute estimate.

Nothing here accepts. The cascade admits a candidate only on the keyed verification, so a probe can
change which angles are tried and in what order, never whether a wrong one is believed. That the false
accept rate is unaffected is measured, not assumed: see tests/test_angle_probe_fpr.py.
"""
import numpy as np
from PIL import Image

__all__ = ["estimate_rotation", "candidate_angles", "has_fill_corners"]

FILL_THRESHOLD = 8.0          # luma below this counts as fill (the rotation pads with black)
MIN_EDGE_POINTS = 8           # too few boundary samples -> the line fit is not trustworthy


def _fill_mask(img):
    return np.asarray(img.convert("L"), np.float64) < FILL_THRESHOLD


def has_fill_corners(img, min_frac=0.005):
    """Whether the frame shows the corner wedges an in-place rotation leaves."""
    m = _fill_mask(img)
    n = m.shape[0] // 4
    corners = m[:n, :n].mean() + m[:n, -n:].mean() + m[-n:, :n].mean() + m[-n:, -n:].mean()
    return bool(corners / 4.0 >= min_frac)


def _edge_slope(mask, from_left=True):
    """Slope of the fill/content boundary over the top quarter, in pixels of x per pixel of y."""
    n = mask.shape[0]
    ys, xs = [], []
    for y in range(0, n // 4):
        row = mask[y] if from_left else mask[y][::-1]
        if not row.any() or row.all():
            continue
        k = int(np.argmax(~row))
        if k == 0 or k > n // 3:
            continue
        ys.append(y); xs.append(k)
    if len(ys) < MIN_EDGE_POINTS:
        return None
    return float(np.polyfit(np.asarray(ys, float), np.asarray(xs, float), 1)[0])


def estimate_magnitudes(img):
    """Candidate rotation MAGNITUDES in degrees from the corner wedges; empty when there are none.

    Both top edges are read. Which of them bounds the wedge that encodes the rotation depends on the
    sign, and the sign is not in this signal, so the other edge returns the complementary angle instead.
    Rather than pick, both readings are returned and the verification settles it -- the estimator says
    what it can measure and nothing more.
    """
    if not has_fill_corners(img):
        return []
    m = _fill_mask(img)
    out = []
    for from_left in (True, False):
        sl = _edge_slope(m, from_left=from_left)
        if sl is None:
            continue
        deg = 90.0 - abs(np.degrees(np.arctan(sl)))
        if 0.2 <= deg <= 89.8:
            out.append(float(deg))
    return sorted({round(d, 2) for d in out})


def estimate_rotation(img):
    """MAGNITUDE of the rotation in degrees from the corner wedges, or None when there are none.

    Only the magnitude is recoverable this way. A rotation by +theta and one by -theta leave mirrored
    wedges of the same area -- measured, the four corner fill fractions agree to within 0.001 -- so the
    sign is simply not in this signal. Rather than guess it, the caller is handed both signs as
    candidates and the keyed verification settles which is right: one extra verification, and no
    reliance on a sign rule that would be wrong half the time.

    The measured quantity is the boundary's inclination from the vertical, so the rotation is its
    complement.
    """
    if not has_fill_corners(img):
        return None
    slope = _edge_slope(_fill_mask(img))
    if slope is None:
        return None
    return float(90.0 - abs(np.degrees(np.arctan(slope))))


def candidate_angles(img, search=180.0, step=10.0, refine=1.0, k=None):
    """Angles to try, best first. Fragment-independent by construction.

    When the wedges give a magnitude, both signs and a fine bracket around each are offered first,
    because a verification that succeeds early saves the rest of the sweep. The exhaustive grid follows,
    so the search stays complete when no estimate is available or the estimate is wrong -- the probe
    orders the work, it does not bound it.
    """
    out = []
    for mag in estimate_magnitudes(img):                   # both edge readings
        for sgn in (-1.0, 1.0):                            # sign is not recoverable; try both
            out.append(sgn * mag)
            for d in np.arange(refine, 3 * refine + 1e-9, refine):
                out += [sgn * mag + d, sgn * mag - d]
    grid = list(np.arange(-search, search + 1e-9, step))
    seen, ranked = set(), []
    for a in out + grid:
        key = round(float(a), 3)
        if key in seen:
            continue
        seen.add(key); ranked.append(float(a))
    return ranked if k is None else ranked[:k]
