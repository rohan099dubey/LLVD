#!/usr/bin/env python3
"""
LLVD — Lattice Layer Vehicle Detection
Entry point script.

Usage:
    python run.py                                        # Run advanced pipeline
    python run.py --config config/default.json
    python run.py --pipeline base                        # Run base pipeline
    python run.py --pipeline optimized                   # Run optimized pipeline
    python run.py --benchmark                            # Compare original vs optimized
    python run.py --help
"""

import argparse
import sys
import os
import time
import json


def main():
    parser = argparse.ArgumentParser(
        description="LLVD — Lattice Layer Vehicle Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py
      Run advanced pipeline with config/default.json

  python run.py --config config/sample_video.json
      Run with a custom config file

  python run.py --pipeline base
      Run the base pipeline (DBSCAN clustering)

  python run.py --pipeline advanced
      Run the advanced pipeline (tracking, counting, speed estimation)

  python run.py --pipeline optimized
      Run the optimized pipeline (Grid Manager add-on with 4-strategy cascade)

  python run.py --benchmark
      Run both original and optimized pipelines, compare results
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.json",
        help="Path to the JSON config file (default: config/default.json)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["advanced", "base", "optimized"],
        default="advanced",
        help="Which pipeline to run (default: advanced)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run both original and optimized pipelines and compare results",
    )
    args = parser.parse_args()

    # Verify config exists
    if not os.path.isfile(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print("  Available configs:")
        config_dir = "config"
        if os.path.isdir(config_dir):
            for f in sorted(os.listdir(config_dir)):
                if f.endswith(".json"):
                    print(f"    config/{f}")
        sys.exit(1)

    # Copy the config to user_input_data.json (expected by the pipeline scripts)
    import shutil
    shutil.copy2(args.config, "user_input_data.json")
    print(f"Using config: {args.config}")

    # Load config for report metadata
    with open(args.config, 'r') as f:
        config_data = json.load(f)

    if args.benchmark:
        print("Starting Benchmark (Original vs. Optimized)...")
        print("=" * 78)
        from src.grid_manager.benchmark import run_benchmark
        run_benchmark()

    elif args.pipeline == "advanced":
        _run_advanced_pipeline(config_data)

    elif args.pipeline == "optimized":
        _run_optimized_pipeline()

    elif args.pipeline == "base":
        _run_base_pipeline(config_data)

    # Cleanup the temporary config copy
    if os.path.isfile("user_input_data.json"):
        os.remove("user_input_data.json")

    print("\nDone! Check the output/ directory for results.")


def _run_advanced_pipeline(config_data):
    """Run advanced pipeline and generate formatted report."""
    print("Starting Advanced Pipeline (tracking + counting + speed)...")
    print("=" * 78)

    from src.advanced_pipeline import main as run_advanced
    from src.advanced_pipeline import (
        timing_stats, merge_profile_logs,
        VIDEO_PATH, COLOR_CHANNEL, NUM_ROWS, NUM_COLS,
        ROI1, ROI2, DEFAULT_BATCH_SIZE, _USE_GRAYSCALE_FASTPATH,
    )
    from multiprocessing import cpu_count
    import psutil
    import numpy as np

    # The original main() prints its own results and returns elapsed time.
    # We capture the data it collects indirectly and generate a clean report.
    execution_time = run_advanced()

    # Generate report from profiling data
    from src.grid_manager.report_writer import PipelineReport
    merged_stats = merge_profile_logs(main_stats=timing_stats)

    report = PipelineReport(pipeline_name="advanced")
    report.set_config(
        video_path=VIDEO_PATH,
        color_channel=COLOR_CHANNEL,
        grid_rows=NUM_ROWS, grid_cols=NUM_COLS,
        roi1=ROI1, roi2=ROI2,
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=max(1, cpu_count() - 1),
        fastpath='grayscale' if _USE_GRAYSCALE_FASTPATH else 'HSV',
    )
    report.set_execution(
        time_s=execution_time,
        memory_mb=psutil.Process().memory_info().rss / (1024 ** 2),
        frames=0,  # not accessible from outside main()
        fps=0,
    )
    report.set_profiling(merged_stats)

    # Save report files (console already printed by the original pipeline)
    report.save()


def _run_optimized_pipeline():
    """Run optimized pipeline — report is handled internally."""
    print("Starting Optimized Pipeline (Grid Manager add-on)...")
    print("=" * 78)
    from src.grid_manager.optimized_pipeline import main as run_optimized
    run_optimized()


def _run_base_pipeline(config_data):
    """Run base pipeline and generate formatted report."""
    print("Starting Base Pipeline (DBSCAN clustering)...")
    print("=" * 78)

    from src.base_pipeline import main as run_base
    import psutil

    run_base()

    # Generate a basic report from config metadata
    from src.grid_manager.report_writer import PipelineReport

    report = PipelineReport(pipeline_name="base")
    report.set_config(
        video_path=config_data.get("video", ""),
        color_channel=config_data.get("color_channel", ""),
        grid_rows=config_data.get("grids", {}).get("rows", 0),
        grid_cols=config_data.get("grids", {}).get("cols", 0),
        batch_size=config_data.get("batch_size", 64),
    )
    report.set_execution(
        memory_mb=psutil.Process().memory_info().rss / (1024 ** 2),
    )

    # Save report files (console already printed by the original pipeline)
    report.save()


if __name__ == "__main__":
    main()
