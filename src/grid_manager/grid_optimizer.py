"""
GridOptimizer — Manages the 4-strategy optimization cascade for the LLVD pipeline.

Strategy 1 (Temporal Sub-sampling):  Runs in MAIN process — skips frames entirely.
Strategy 4+2 (Pixel Activity Check): Runs in WORKER process — parallel-safe pixel delta.
Strategy 3 (Sentinel Boundary Check): Runs in WORKER process — scans entry rows only.

The GridOptimizer instance lives in the main process and holds cached results
for skipped frames. Worker-side strategies are @staticmethod (no shared state).
"""

import cv2
import numpy as np


class GridOptimizer:
    """
    Stateful optimizer that lives in the main process.
    Manages temporal sub-sampling and caches results for skipped frames.
    Worker-callable strategies are static methods (parallel-safe).
    """

    def __init__(self, total_cells, num_rows, num_cols,
                 subsample_interval=5,
                 min_pixel_activity=100, entry_rows=None,
                 sentinel_threshold=50):
        """
        Args:
            total_cells: Total grid cells (rows * cols) for density calculation.
            num_rows: Number of grid rows (needed to size cached matrices).
            num_cols: Number of grid columns (needed to size cached matrices).
            subsample_interval: Process every Nth frame (Strategy 1). Default: 5.
            min_pixel_activity: Minimum changed pixels to trigger processing (Strategy 4+2).
            entry_rows: List of row indices where vehicles enter (Strategy 3).
                        Use negative indices for bottom rows, e.g. [0, -1].
            sentinel_threshold: Pixel count threshold for entry row activity (Strategy 3).
        """
        self.total_cells = total_cells
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.subsample_interval = subsample_interval
        self.min_pixel_activity = min_pixel_activity
        self.entry_rows = entry_rows if entry_rows is not None else [0, -1]
        self.sentinel_threshold = sentinel_threshold

        # Cached results for skipped frames — correctly sized from the start
        self.last_mat1 = np.zeros((num_rows, num_cols), dtype=np.int8)
        self.last_mat2 = np.zeros((num_rows, num_cols), dtype=np.int8)
        self.last_d1 = 0.0
        self.last_d2 = 0.0
        self.last_comps1 = []
        self.last_comps2 = []

        # Performance stats
        self.stats = {
            'skipped_temporal': 0,
            'skipped_pixel': 0,
            'skipped_sentinel': 0,
            'computed': 0,
        }

    # ──────────────────────────────────────────────────────────
    # Strategy 1: Temporal Sub-sampling (MAIN PROCESS)
    # ──────────────────────────────────────────────────────────

    def should_skip_temporal(self, frame_idx):
        """
        Returns True if this frame should be skipped based on temporal interval.
        Only every Nth frame gets dispatched to the worker pool.
        Never skips the first frame (idx <= 1) to prime the cache.
        """
        if frame_idx <= 1:
            return False  # Always compute the first frame to prime cache
        return frame_idx % self.subsample_interval != 0

    # ──────────────────────────────────────────────────────────
    # Cache management (MAIN PROCESS)
    # ──────────────────────────────────────────────────────────

    def get_cached_result(self):
        """Return last known results for skipped frames."""
        return (self.last_mat1, self.last_mat2,
                self.last_d1, self.last_d2,
                self.last_comps1, self.last_comps2)

    def update_cache(self, mat1, mat2, d1, d2, comps1, comps2):
        """Store computed results for future skipped frames."""
        self.last_mat1 = mat1
        self.last_mat2 = mat2
        self.last_d1 = d1
        self.last_d2 = d2
        self.last_comps1 = comps1
        self.last_comps2 = comps2

    # ──────────────────────────────────────────────────────────
    # Strategy 4+2: Pixel Activity Check (WORKER PROCESS)
    # Parallel-safe — uses raw pixels, not prior grid state
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def should_skip_pixel_activity(roi_f1_gray, roi_f2_gray, threshold=100):
        """
        Check if there's enough pixel-level change between two frames.
        If the raw pixel diff is below threshold, the grid result won't
        change meaningfully — skip full computation.

        This merges the original Strategy 2 (Grid Delta) and Strategy 4
        (Pixel Activity) into a single parallel-safe check.

        Args:
            roi_f1_gray: Grayscale ROI from frame N.
            roi_f2_gray: Grayscale ROI from frame N+1.
            threshold: Minimum non-zero pixels in the diff to trigger processing.

        Returns:
            True if the check says "skip" (not enough motion).
        """
        diff = cv2.absdiff(roi_f1_gray, roi_f2_gray)
        # Apply a small threshold to ignore sensor noise
        _, diff_bin = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        return cv2.countNonZero(diff_bin) < threshold

    # ──────────────────────────────────────────────────────────
    # Strategy 3: Sentinel Boundary Check (WORKER PROCESS)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def should_skip_sentinel(roi_diff_binary, grid_h, num_rows,
                             entry_rows, threshold=50):
        """
        Only scan the entry rows of the grid. If no motion is detected
        in entry rows AND previous grid was empty, no new vehicles can
        have entered — skip full grid computation.

        Args:
            roi_diff_binary: Thresholded binary diff of the ROI.
            grid_h: Height of each grid cell in pixels.
            num_rows: Total number of grid rows.
            entry_rows: List of row indices (supports negative indexing).
            threshold: Minimum non-zero pixels in entry strip to detect motion.

        Returns:
            True if the check says "skip" (no motion in entry rows).
        """
        for row_idx in entry_rows:
            # Resolve negative indices
            actual_row = row_idx if row_idx >= 0 else num_rows + row_idx
            if actual_row < 0 or actual_row >= num_rows:
                continue

            y_start = actual_row * grid_h
            y_end = min(y_start + grid_h, roi_diff_binary.shape[0])
            entry_strip = roi_diff_binary[y_start:y_end, :]

            if cv2.countNonZero(entry_strip) > threshold:
                return False  # Motion detected in entry row → can't skip

        return True  # No motion in any entry row → safe to skip

    # ──────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────

    def record_skip(self, reason):
        """Record a skip event. reason: 'temporal', 'pixel', or 'sentinel'."""
        key = f'skipped_{reason}'
        if key in self.stats:
            self.stats[key] += 1

    def record_compute(self):
        """Record a full computation event."""
        self.stats['computed'] += 1

    def get_stats(self):
        """Return optimization stats with computed percentages."""
        total = sum(self.stats.values())
        if total == 0:
            return self.stats, {}

        pcts = {k: v / total * 100 for k, v in self.stats.items()}
        return self.stats, pcts

    def print_stats(self):
        """Print a formatted summary of optimization performance."""
        counts, pcts = self.get_stats()
        total = sum(counts.values())
        if total == 0:
            return

        print(f"\n{'─' * 70}")
        print("GRID MANAGER — OPTIMIZATION STATS")
        print(f"{'─' * 70}")
        for key in ['skipped_temporal', 'skipped_pixel', 'skipped_sentinel', 'computed']:
            count = counts.get(key, 0)
            pct = pcts.get(key, 0)
            label = key.replace('skipped_', 'Skip: ').replace('computed', 'Full Compute')
            print(f"  {label:<25} {count:>6} frames ({pct:>5.1f}%)")
        skip_total = total - counts.get('computed', 0)
        print(f"  {'─' * 40}")
        print(f"  {'Total Frames':<25} {total:>6}")
        print(f"  {'Total Skipped':<25} {skip_total:>6} ({skip_total / total * 100:.1f}%)")
        print(f"{'─' * 70}")
