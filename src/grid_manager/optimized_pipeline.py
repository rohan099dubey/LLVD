"""
Optimized Pipeline — Wraps the original advanced_pipeline with the
4-strategy Grid Manager optimization cascade.

This file imports ALL processing functions from the original pipeline
and rewires only the main processing loop to add:
  - Strategy 1: Temporal sub-sampling in main process (before pool.map)
  - Strategy 3 & 4+2: Pixel/Sentinel checks in worker processes

The original advanced_pipeline.py is NEVER modified.
"""

import cv2
import numpy as np
import psutil
import time
import os
import json
import sys
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Import EVERYTHING from the original pipeline (zero modifications)
# ──────────────────────────────────────────────────────────────
from src.advanced_pipeline import (
    # Config values
    VIDEO_PATH, COLOR_CHANNEL, NUM_ROWS, NUM_COLS, TOTAL_CELLS,
    ROI1, ROI2, GRID_W1, GRID_H1, GRID_W2, GRID_H2,
    ASSUMED_FPS, DEFAULT_BATCH_SIZE,
    _USE_GRAYSCALE_FASTPATH,
    # Processing functions (used by worker via its own import)
    process_single_roi, find_connected_components,
    classify_scene, adaptive_equalize,
    # I/O and visualization
    _load_image_sequence, _crop_rois,
    _equalize_frame_rois_adaptive,
    batch_roi_generator, prefetch_batches,
    AsyncVideoWriter,
    draw_grid_overlay, draw_tracking_info, draw_scene_info,
    # Tracking
    CentroidTracker,
    # Profiling
    timing_stats, merge_profile_logs, print_profiling_results,
    LOG_DIR,
)

from src.grid_manager.grid_optimizer import GridOptimizer
from src.grid_manager.optimized_worker import optimized_worker_task


def load_optimization_config():
    """Load optimization params from config, with sensible defaults."""
    try:
        with open("user_input_data.json", "r") as f:
            config = json.load(f)
        opt = config.get("optimization", {})
    except (FileNotFoundError, json.JSONDecodeError):
        opt = {}

    return {
        'subsample_interval': int(opt.get('subsample_interval', 5)),
        'min_pixel_activity': int(opt.get('min_pixel_activity', 100)),
        'entry_rows': list(opt.get('entry_rows', [0, -1])),
        'sentinel_threshold': int(opt.get('sentinel_threshold', 50)),
        'enabled': bool(opt.get('enabled', True)),
    }


def main():
    """
    Optimized version of advanced_pipeline.main().
    Same I/O, same output format, but with Grid Manager optimizations.
    """
    start = time.time()
    opt_config = load_optimization_config()

    # Clean old logs
    for fp in LOG_DIR.glob("prof_*.json"):
        try:
            fp.unlink()
        except Exception:
            pass

    # ── Video setup (same as original) ──
    is_seq = os.path.isdir(VIDEO_PATH)
    if is_seq:
        files = _load_image_sequence(VIDEO_PATH)
        first = cv2.imread(files[0])
        h, w = first.shape[:2]
        total_frames = len(files) - 1
        fps = ASSUMED_FPS
        del first
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        fps = cap.get(cv2.CAP_PROP_FPS) or ASSUMED_FPS
        cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs('output/videos', exist_ok=True)
    writer = AsyncVideoWriter('output/videos/output_optimized.mp4', fourcc, fps, (w, h))

    # ── Initial scene check (same as original) ──
    if is_seq:
        first = cv2.imread(_load_image_sequence(VIDEO_PATH)[0])
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        _, first = cap.read()
        cap.release()
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    init_lighting, init_foggy, init_bright, init_fog_idx = classify_scene(first_gray)
    del first, first_gray

    # ── Initialize trackers and optimizer ──
    tracker1 = CentroidTracker()
    tracker2 = CentroidTracker()
    batch_size = DEFAULT_BATCH_SIZE
    n_procs = max(1, cpu_count() - 1)

    optimizer = GridOptimizer(
        total_cells=TOTAL_CELLS,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS,
        subsample_interval=opt_config['subsample_interval'],
        min_pixel_activity=opt_config['min_pixel_activity'],
        entry_rows=opt_config['entry_rows'],
        sentinel_threshold=opt_config['sentinel_threshold'],
    )

    print(f"Pipeline: OPTIMIZED | {n_procs} workers | batch={batch_size}")
    print(f"Scene: {init_lighting.upper()}"
          f"{' + FOG' if init_foggy else ''}"
          f" (bright={init_bright:.0f}, fog_idx={init_fog_idx:.2f})")
    print(f"Features: Tracking | Counting | Speed | Night/Fog Adapt")
    print(f"Grid: {NUM_ROWS}x{NUM_COLS} | "
          f"Fast-path: {'grayscale' if _USE_GRAYSCALE_FASTPATH else 'HSV'}")
    print(f"Optimization: subsample={opt_config['subsample_interval']} | "
          f"pixel_thresh={opt_config['min_pixel_activity']} | "
          f"entry_rows={opt_config['entry_rows']}")

    d_lane1, d_lane2 = [], []
    speed_samples_l1, speed_samples_l2 = [], []
    scene_log = []
    frame_count = 0

    # ── Main processing loop with optimizations ──
    with Pool(processes=n_procs) as pool:
        pbar = tqdm(total=total_frames, desc="Processing (optimized)")
        gen = batch_roi_generator(VIDEO_PATH, batch_size)

        for batch in prefetch_batches(gen):
            batch_lighting, batch_foggy = batch[0][6], batch[0][7]
            scene_log.append((batch_lighting, batch_foggy))

            # Separate frames into tasks vs skipped
            worker_tasks = []
            skipped_items = []

            for item in batch:
                frame_idx = item[0]
                frame_for_viz = item[5]  # original frame for drawing

                # ── STRATEGY 1: Temporal Sub-sampling (Main Process) ──
                if opt_config['enabled'] and optimizer.should_skip_temporal(frame_idx):
                    optimizer.record_skip('temporal')
                    cached = optimizer.get_cached_result()
                    skipped_items.append((frame_idx, frame_for_viz, cached))
                    continue

                # Pack args for optimized worker (original 7 args + 3 optimization params)
                worker_tasks.append((
                    frame_idx,
                    item[1], item[2],  # r1f1, r1f2
                    item[3], item[4],  # r2f1, r2f2
                    item[6], item[7],  # lighting, is_foggy
                    opt_config['min_pixel_activity'],
                    opt_config['entry_rows'],
                    opt_config['sentinel_threshold'],
                ))

            # ── Dispatch non-skipped frames to worker pool ──
            computed_results = {}
            if worker_tasks:
                ipc_start = time.perf_counter()
                raw_results = pool.map(optimized_worker_task, worker_tasks)
                ipc_elapsed = time.perf_counter() - ipc_start

                # Record IPC timing
                s = timing_stats['pool_map_ipc']
                s['total_time'] += ipc_elapsed
                s['call_count'] += 1
                s['min_time'] = min(s['min_time'], ipc_elapsed)
                s['max_time'] = max(s['max_time'], ipc_elapsed)

                for fidx, r1_result, r2_result in raw_results:
                    computed_results[fidx] = (r1_result, r2_result)

            # ── Build the original_frames map for visualization ──
            original_frames = {item[0]: item[5] for item in batch}

            # ── Process results in frame order ──
            all_frame_indices = sorted(
                [item[0] for item in batch]
            )

            for fidx in all_frame_indices:
                frame = original_frames[fidx]

                if fidx in computed_results:
                    r1_res, r2_res = computed_results[fidx]

                    # Handle per-ROI results (may be skipped at worker level)
                    mat1 = _resolve_roi_result(r1_res, optimizer, 'roi1')
                    mat2 = _resolve_roi_result(r2_res, optimizer, 'roi2')

                    d1 = r1_res['d'] if r1_res['status'] == 'computed' else optimizer.last_d1
                    d2 = r2_res['d'] if r2_res['status'] == 'computed' else optimizer.last_d2
                    comps1 = r1_res['comps'] if r1_res['status'] == 'computed' else optimizer.last_comps1
                    comps2 = r2_res['comps'] if r2_res['status'] == 'computed' else optimizer.last_comps2

                    # Track skip stats from workers
                    for res in [r1_res, r2_res]:
                        if res['status'] == 'skip_pixel':
                            optimizer.record_skip('pixel')
                        elif res['status'] == 'skip_sentinel':
                            optimizer.record_skip('sentinel')
                        elif res['status'] == 'computed':
                            optimizer.record_compute()

                    # Update cache with latest computed values
                    if r1_res['status'] == 'computed' or r2_res['status'] == 'computed':
                        optimizer.update_cache(mat1, mat2, d1, d2, comps1, comps2)

                else:
                    # This was a temporally skipped frame — use cached data
                    _, _, d1, d2, comps1, comps2 = optimizer.get_cached_result()
                    mat1 = optimizer.last_mat1
                    mat2 = optimizer.last_mat2

                # ── Visualization (same as original) ──
                _equalize_frame_rois_adaptive(frame, batch_lighting)

                draw_grid_overlay(frame, ROI1[0], ROI1[1], GRID_W1, GRID_H1, mat1)
                draw_grid_overlay(frame, ROI2[0], ROI2[1], GRID_W2, GRID_H2, mat2)

                tracker1.update(comps1, GRID_W1, GRID_H1, ROI1[0], ROI1[1])
                tracker2.update(comps2, GRID_W2, GRID_H2, ROI2[0], ROI2[1])

                tracker1.check_counting_line(ROI1[1], ROI1[3])
                tracker2.check_counting_line(ROI2[1], ROI2[3])

                draw_tracking_info(frame, tracker1, ROI1[0], ROI1[1], ROI1[3], "L1")
                draw_tracking_info(frame, tracker2, ROI2[0], ROI2[1], ROI2[3], "L2")

                draw_scene_info(frame, batch_lighting, batch_foggy, init_bright, init_fog_idx)
                cv2.putText(frame, f"Frame: {fidx}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # "OPT" indicator
                cv2.putText(frame, "OPT", (w - 60, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                d_lane1.append(d1)
                d_lane2.append(d2)
                for spd in tracker1.speeds.values():
                    if spd > 1.0:
                        speed_samples_l1.append(spd)
                for spd in tracker2.speeds.values():
                    if spd > 1.0:
                        speed_samples_l2.append(spd)

                writer.write(frame)
                frame_count += 1

            pbar.update(len(batch))
            del batch, worker_tasks, skipped_items, original_frames, computed_results

        pbar.close()

    writer.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    # ── Generate report using PipelineReport ──
    from src.grid_manager.report_writer import PipelineReport

    elapsed = time.time() - start
    mem_mb = psutil.Process().memory_info().rss / (1024 ** 2)

    avg_speed_l1 = np.mean(speed_samples_l1) if speed_samples_l1 else 0.0
    avg_speed_l2 = np.mean(speed_samples_l2) if speed_samples_l2 else 0.0

    scene_counts = defaultdict(int)
    for lit, fog in scene_log:
        key = lit + ('+fog' if fog else '')
        scene_counts[key] += 1

    report = PipelineReport(pipeline_name="optimized")
    report.set_config(
        video_path=VIDEO_PATH,
        color_channel=COLOR_CHANNEL,
        grid_rows=NUM_ROWS, grid_cols=NUM_COLS,
        roi1=ROI1, roi2=ROI2,
        batch_size=batch_size,
        num_workers=n_procs,
        fastpath='grayscale' if _USE_GRAYSCALE_FASTPATH else 'HSV',
    )
    report.set_execution(
        time_s=elapsed, memory_mb=mem_mb,
        frames=frame_count, fps=frame_count / elapsed if elapsed > 0 else 0,
    )
    report.set_density(
        lane1_avg=float(np.mean(d_lane1)) if d_lane1 else 0.0,
        lane2_avg=float(np.mean(d_lane2)) if d_lane2 else 0.0,
        lane1_values=d_lane1, lane2_values=d_lane2,
    )
    report.set_counting(lane1=tracker1.total_count, lane2=tracker2.total_count)
    report.set_speed(
        lane1_avg=float(avg_speed_l1), lane2_avg=float(avg_speed_l2),
        lane1_samples=len(speed_samples_l1), lane2_samples=len(speed_samples_l2),
    )
    report.set_scene(
        scene_counts=scene_counts,
        initial_lighting=init_lighting, initial_foggy=init_foggy,
        brightness=init_bright, fog_index=init_fog_idx,
    )
    report.set_tracking(
        total_ids=tracker1.next_id + tracker2.next_id,
        active_l1=len(tracker1.objects), active_l2=len(tracker2.objects),
    )
    report.set_optimization(stats=optimizer.stats, config=opt_config)

    # Merge profiling and add to report
    merged = merge_profile_logs(main_stats=timing_stats)
    report.set_profiling(merged)

    report.print_console()
    report.save()

    return elapsed


def _resolve_roi_result(roi_result, optimizer, roi_name):
    """Extract the grid matrix from a worker result, falling back to cache."""
    if roi_result['status'] == 'computed':
        return roi_result['mat']
    elif roi_name == 'roi1':
        return optimizer.last_mat1
    else:
        return optimizer.last_mat2


if __name__ == '__main__':
    print(f"\n{'=' * 78}")
    print("  OPTIMIZED GRID MANAGER — TRAFFIC ANALYSIS")
    print(f"{'=' * 78}\n")
    main()
