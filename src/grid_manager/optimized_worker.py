"""
Optimized Worker — Parallel-safe worker_task with Strategy 3 & 4+2 cascade.

This replaces the original worker_task for the optimized pipeline.
It wraps the original process_single_roi with early-exit checks.
"""

import cv2
import numpy as np

# These imports happen at module load time when the worker process starts.
# multiprocessing.Pool forks/spawns, so the original module is loaded in each worker.
from src.advanced_pipeline import (
    process_single_roi, find_connected_components,
    GRID_W1, GRID_H1, GRID_W2, GRID_H2,
    NUM_ROWS, NUM_COLS, TOTAL_CELLS
)
from src.grid_manager.grid_optimizer import GridOptimizer


def optimized_worker_task(args):
    """
    Optimized version of worker_task with Strategy 3 & 4+2 cascade.

    The cascade per ROI:
      1. Pixel Activity Check (Strategy 4+2) — if diff is tiny, skip.
      2. Sentinel Boundary Check (Strategy 3)  — if entry rows are quiet, skip.
      3. Full Computation — original process_single_roi + find_connected_components.

    Args:
        args: Tuple of:
            frame_idx, r1f1, r1f2, r2f1, r2f2, lighting, is_foggy,
            pixel_threshold, entry_rows, sentinel_threshold

    Returns:
        Tuple of (frame_idx, roi1_result, roi2_result)
        Each roi_result is a dict with keys:
            'status': 'computed' | 'skip_pixel' | 'skip_sentinel'
            'mat':    grid matrix (or None if skipped)
            'd':      density float (or 0.0 if skipped)
            'comps':  components list (or [] if skipped)
    """
    (frame_idx, r1f1, r1f2, r2f1, r2f2,
     lighting, is_foggy,
     pixel_threshold, entry_rows, sentinel_threshold) = args

    roi_inputs = [
        (r1f1, r1f2, GRID_W1, GRID_H1),
        (r2f1, r2f2, GRID_W2, GRID_H2),
    ]

    results = []

    for roi_f1, roi_f2, grid_w, grid_h in roi_inputs:
        # Convert to grayscale for the optimization checks
        g1 = cv2.cvtColor(roi_f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(roi_f2, cv2.COLOR_BGR2GRAY)

        # ── Strategy 4+2: Pixel Activity Check (parallel-safe) ──
        if GridOptimizer.should_skip_pixel_activity(g1, g2, pixel_threshold):
            results.append({
                'status': 'skip_pixel',
                'mat': None,
                'd': 0.0,
                'comps': [],
            })
            continue

        # ── Strategy 3: Sentinel Boundary Check ──
        diff = cv2.absdiff(g1, g2)
        _, diff_bin = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        if GridOptimizer.should_skip_sentinel(
            diff_bin, grid_h, NUM_ROWS, entry_rows, sentinel_threshold
        ):
            results.append({
                'status': 'skip_sentinel',
                'mat': None,
                'd': 0.0,
                'comps': [],
            })
            continue

        # ── Full Computation (all checks failed) ──
        mat = process_single_roi(roi_f1, roi_f2, grid_w, grid_h, lighting, is_foggy)
        d = float(np.count_nonzero(mat)) / TOTAL_CELLS

        comps = find_connected_components(mat)
        comps_data = [
            {'centroid': c['centroid'], 'area': c['area'], 'bbox': c['bbox']}
            for c in comps
        ]

        results.append({
            'status': 'computed',
            'mat': mat,
            'd': d,
            'comps': comps_data,
        })

    return frame_idx, results[0], results[1]
