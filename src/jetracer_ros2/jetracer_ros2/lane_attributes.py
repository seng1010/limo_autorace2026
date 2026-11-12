"""Image-based lane attributes for diagnostics, not mission decisions."""

import cv2
import numpy as np


def classify_lane(bird, fit, observed_y, config, zigzag=False):
    """Classify only a measured boundary; never label a reconstructed boundary.

    Gaps are measured along the fitted line inside its observed vertical span.
    Dashed is a heuristic: shadows/occlusion can also produce repeated gaps.
    """
    if fit is None or len(observed_y) == 0:
        return 'unobserved'
    lo, hi = int(np.min(observed_y)), int(np.max(observed_y))
    if hi - lo < bird.shape[0] * 0.30:
        return 'unknown/unknown/unknown'
    hls = cv2.cvtColor(bird[lo:hi + 1], cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    yellow = ((h >= config.yellow_hue_min) & (h <= config.yellow_hue_max)
              & (s >= config.yellow_saturation_min)
              & (l >= config.yellow_lightness_min))
    white = ((l >= config.white_lightness_min)
             & (s <= config.white_saturation_max) & ~yellow)
    ys = np.arange(lo, hi + 1)
    centers = np.polyval(fit, ys)
    corridor = np.abs(np.arange(bird.shape[1])[None, :] - centers[:, None]) <= 12
    yellow &= corridor
    white &= corridor
    counts = [int(yellow.sum()), int(white.sum())]
    winner = int(np.argmax(counts))
    if max(counts) < 60 or max(counts) / max(1, sum(counts)) < 0.70:
        color, pattern = 'unknown', 'unknown'
    else:
        color = ('yellow', 'white')[winner]
        rows = ((yellow if winner == 0 else white).sum(axis=1) >= 2)
        # Close tiny holes only; retain substantial gaps between painted dashes.
        rows = cv2.morphologyEx(rows.astype(np.uint8)[:, None], cv2.MORPH_CLOSE,
                                np.ones((5, 1), np.uint8)).ravel().astype(bool)
        occupied = np.flatnonzero(rows)
        if occupied.size == 0:
            pattern = 'unknown'
        else:
            rows = rows[occupied[0]:occupied[-1] + 1]
            edges = np.flatnonzero(np.diff(np.r_[True, rows, True].astype(int)))
            gaps = edges[1::2] - edges[::2]
            substantial_gaps = int(np.count_nonzero(gaps >= 12))
            if substantial_gaps >= 2 and rows.mean() < 0.85:
                pattern = 'dashed'
            elif rows.mean() >= 0.90:
                pattern = 'solid'
            else:
                pattern = 'unknown'
    slopes = np.polyval(np.polyder(fit), np.linspace(lo, hi, 24))
    heading_span = float(np.ptp(np.arctan(slopes)))
    shape = 'zigzag' if zigzag else ('curve' if heading_span >= 0.20 else 'straight')
    return f'{color}/{pattern}/{shape}'


def scale_steering(target, gain):
    """Increase steering demand without increasing the normalized angle limit."""
    if not np.isfinite(gain) or gain <= 0:
        gain = 1.0
    return float(np.clip(target * gain, -1.0, 1.0))
