# Grid Optimization Add-On — Implementation Plan

## Goal

Implement the 4 optimization strategies from [strategy.md](file:///e:/WORK/additionalProject/LLVD/response/gemini/strategy.md) as a **completely separated add-on** that wraps the original pipeline without modifying any of the colleague's code. Your code lives in `src/addons/`, the original stays untouched in `src/`.

---

## Proposed Changes

### Add-On Module (`src/addons/`)

#### [NEW] [__init__.py](file:///e:/WORK/additionalProject/LLVD/src/addons/__init__.py)
Empty init for the addons package.

---

#### [NEW] [grid_optimizer.py](file:///e:/WORK/additionalProject/LLVD/src/addons/grid_optimizer.py)

The core optimization engine. Contains:

- **`GridOptimizer` class** — stateful per-ROI optimizer that implements all 4 strategies:

```python
class GridOptimizer:
    def __init__(self, total_cells, subsample_interval=5, min_pixel_activity=100,
                 entry_rows=(0, -1)):
        self.prev_grid = None         # Cache for Strategy 2 (Delta)
        self.prev_density = 0.0       # Cached density
        self.prev_comps = []          # Cached components
        self.total_cells = total_cells
        self.subsample_interval = subsample_interval
        self.min_pixel_activity = min_pixel_activity
        self.entry_rows = entry_rows  # Rows where vehicles can enter

        # Stats tracking
        self.stats = {'skipped_temporal': 0, 'skipped_pixel': 0,
                      'skipped_delta': 0, 'skipped_sentinel': 0,
                      'computed': 0}
```

- **`should_skip_temporal(frame_idx)`** — Strategy 1: returns `True` if `frame_idx % interval != 0`
- **`should_skip_pixel_activity(roi_diff_gray)`** — Strategy 4: returns `True` if `cv2.countNonZero(roi_diff_gray) < min_pixel_activity`
- **`should_skip_sentinel(roi_diff_gray, grid_h)`** — Strategy 3: checks only entry rows for motion
- **`should_skip_delta(current_grid)`** — Strategy 2: returns `True` if `np.array_equal(current_grid, self.prev_grid)`
- **`get_cached_result()`** — returns [(prev_grid, prev_density, prev_comps)](file:///e:/WORK/additionalProject/LLVD/src/base_pipeline.py#290-443)
- **`update_cache(grid, density, comps)`** — stores the new computed state
- **`get_stats()`** — returns optimization hit rates

---

#### [NEW] [optimized_worker.py](file:///e:/WORK/additionalProject/LLVD/src/addons/optimized_worker.py)

A replacement [worker_task](file:///e:/WORK/additionalProject/LLVD/src/advanced_pipeline.py#435-451) that wraps the original pipeline functions with optimization checks:

```python
def optimized_worker_task(args):
    """
    Wraps the original worker_task with the 4-strategy optimization cascade.
    Falls through to the original process_single_roi only when all checks fail.
    """
    frame_idx, r1f1, r1f2, r2f1, r2f2, lighting, is_foggy, opt_state_roi1, opt_state_roi2 = args

    # For each ROI, run the optimization cascade:
    # 1. Temporal sub-sampling check
    # 2. Global pixel activity check (early exit)
    # 3. Sentinel boundary check
    # 4. If all pass → call original process_single_roi → delta check
    # 5. Update cache
```

> [!IMPORTANT]
> **Multiprocessing challenge**: `GridOptimizer` objects are stateful and live in the main process. Worker processes can't share state directly. The solution is to pass the prior cached state as serialized args and return the new state from the worker, with main process managing the `GridOptimizer` instances.

The actual flow:
1. **Main process** holds 2 `GridOptimizer` instances (one per ROI)
2. Before dispatching each frame to the pool, main checks Strategy 1 (temporal) — skip entirely if not a "compute frame"
3. For compute frames, the worker receives the raw ROI diff alongside the normal args
4. Worker checks Strategy 4 (pixel activity) and Strategy 3 (sentinel) — early exits return a "skip" flag
5. If not skipped, worker calls [process_single_roi](file:///e:/WORK/additionalProject/LLVD-latest/rparallel_imp%20%281%29.py#414-433) → gets grid → worker checks Strategy 2 (delta) by comparing against the previous grid passed as an arg
6. Worker returns [(frame_idx, mat, density, comps, was_skipped)](file:///e:/WORK/additionalProject/LLVD/src/base_pipeline.py#290-443) 
7. Main process updates `GridOptimizer` cache

---

#### [NEW] [optimized_pipeline.py](file:///e:/WORK/additionalProject/LLVD/src/addons/optimized_pipeline.py)

A new [main()](file:///e:/WORK/additionalProject/LLVD-latest/honors_imp%20%281%29.py#445-448) function that imports everything from the original `advanced_pipeline` and rewires the processing loop:

```python
# Import ALL original functions (zero modifications to original code)
from src.advanced_pipeline import (
    process_single_roi, find_connected_components,
    batch_roi_generator, prefetch_batches, CentroidTracker,
    # ... all draw/viz functions, config values, etc.
)
from src.addons.grid_optimizer import GridOptimizer
```

- Replaces only the inner processing loop (`pool.map` call + result handling)
- Adds optimization stats to the output summary
- Output goes to `output/videos/output_optimized.mp4`

---

#### [NEW] [benchmark.py](file:///e:/WORK/additionalProject/LLVD/src/addons/benchmark.py)

Side-by-side comparison tool:
- Runs the original pipeline, records time + density values
- Runs the optimized pipeline on the same video
- Prints a comparison table: time, FPS, total counts, density diff, optimization hit rates

---

### Config Update

#### [MODIFY] [default.json](file:///e:/WORK/additionalProject/LLVD/config/default.json)

Add optimization parameters:

```json
{
  "optimization": {
    "subsample_interval": 5,
    "min_pixel_activity": 100,
    "entry_rows": [0, 6],
    "enabled": true
  }
}
```

---

### Entry Point Update

#### [MODIFY] [run.py](file:///e:/WORK/additionalProject/LLVD/run.py)

Add a third pipeline option:

```bash
python run.py --pipeline optimized    # NEW: runs optimized add-on
python run.py --pipeline advanced     # Original (unchanged)
python run.py --pipeline base         # Base (unchanged)
python run.py --benchmark             # Runs both and compares
```

---

## Final Folder Structure

```
LLVD/
├── run.py                          # Updated: adds --pipeline optimized + --benchmark
├── src/
│   ├── advanced_pipeline.py        # ⛔ UNTOUCHED — colleague's code
│   ├── base_pipeline.py            # ⛔ UNTOUCHED — colleague's code
│   └── addons/                     # ✅ YOUR CODE — fully separated
│       ├── __init__.py
│       ├── grid_optimizer.py       # GridOptimizer class (4 strategies)
│       ├── optimized_worker.py     # Optimized worker_task wrapper
│       ├── optimized_pipeline.py   # New main() with optimization loop
│       └── benchmark.py            # Side-by-side comparison tool
└── config/
    └── default.json                # Updated: optimization params added
```

---

## Verification Plan

### Automated Test — Benchmark Comparison

```bash
cd e:/WORK/additionalProject/LLVD
python run.py --benchmark
```

This will:
1. Run the **original** `advanced_pipeline` on MVI_39761 → record execution time, density arrays, vehicle counts
2. Run the **optimized** pipeline on the same data → record the same metrics
3. Print a comparison table showing:
   - Execution time speedup (optimized should be faster)
   - Density accuracy difference (should be < 5% RMSE since we're skipping similar frames)
   - Vehicle count difference (should be very close)
   - Per-strategy skip rates (temporal, pixel, sentinel, delta)

**Expected results:**
- **Speed**: 30-60% faster (depending on traffic density in the video)
- **Accuracy**: < 5% density deviation from original (acceptable for monitoring)
- **Counts**: Within ±2 vehicles of original (temporal sub-sampling may miss fast crossings)

### Manual Verification

1. Run `python run.py --pipeline optimized` and check `output/videos/output_optimized.mp4`
2. Compare a few frames visually against `output/videos/output_advanced_cv.mp4` (from `--pipeline advanced`)
3. Verify the optimization stats printed to console show non-zero skip rates

> [!NOTE]
> Temporal sub-sampling (Strategy 1) will intentionally reduce accuracy slightly in exchange for speed. The `subsample_interval` config parameter lets you tune this tradeoff (lower = more accurate, higher = faster).
