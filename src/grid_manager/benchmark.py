"""
Benchmark — Runs all 3 pipelines (base, advanced, optimized) and produces
a comprehensive side-by-side comparison of all stats.

Usage:
    python run.py --benchmark
    python -m src.grid_manager.benchmark
"""

import io
import re
import sys
import time
import json
import psutil
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout

from src.grid_manager.report_writer import REPORT_DIR


# ── Stdout capture utility ──────────────────────────────────────

class TeeCapture:
    """Captures stdout while still printing to the real terminal."""
    def __init__(self, real_stdout):
        self.real = real_stdout
        self.buffer = io.StringIO()

    def write(self, text):
        self.real.write(text)
        self.buffer.write(text)

    def flush(self):
        self.real.flush()

    def getvalue(self):
        return self.buffer.getvalue()


# ── Parsers: extract stats from each pipeline's console output ──

def _parse_advanced_output(text):
    """Parse stats from the advanced pipeline's printed output."""
    data = {}
    # Execution
    m = re.search(r'Execution Time:\s+([\d.]+)s', text)
    if m: data['time_s'] = float(m.group(1))
    m = re.search(r'Memory Usage:\s+([\d.]+)\s*MB', text)
    if m: data['memory_mb'] = float(m.group(1))
    m = re.search(r'Frames Processed:\s+(\d+)', text)
    if m: data['frames'] = int(m.group(1))
    m = re.search(r'Processing Speed:\s+([\d.]+)\s*FPS', text)
    if m: data['fps'] = float(m.group(1))
    # Density
    m = re.search(r'Avg Lane 1:\s+([\d.]+)', text)
    if m: data['density_l1'] = float(m.group(1))
    m = re.search(r'Avg Lane 2:\s+([\d.]+)', text)
    if m: data['density_l2'] = float(m.group(1))
    # Counting
    m = re.search(r'Lane 1 Total:\s+(\d+)', text)
    if m: data['count_l1'] = int(m.group(1))
    m = re.search(r'Lane 2 Total:\s+(\d+)', text)
    if m: data['count_l2'] = int(m.group(1))
    m = re.search(r'Combined:\s+(\d+)', text)
    if m: data['count_total'] = int(m.group(1))
    # Speed
    m = re.search(r'Lane 1 Avg:\s+([\d.]+)\s*km/h', text)
    if m: data['speed_l1'] = float(m.group(1))
    m = re.search(r'Lane 2 Avg:\s+([\d.]+)\s*km/h', text)
    if m: data['speed_l2'] = float(m.group(1))
    # Tracking
    m = re.search(r'Total IDs assigned:\s+(\d+)', text)
    if m: data['track_ids'] = int(m.group(1))
    return data


def _parse_base_output(text):
    """Parse stats from the base pipeline's printed output."""
    data = {}
    m = re.search(r'Execution time:\s+([\d.]+)s', text)
    if m: data['time_s'] = float(m.group(1))
    m = re.search(r'Processing speed.*?:\s+([\d.]+)', text)
    if m: data['fps'] = float(m.group(1))
    m = re.search(r'Memory:\s+([\d.]+)\s*MB', text)
    if m: data['memory_mb'] = float(m.group(1))
    m = re.search(r'Lane 1:.*?density:\s+([\d.]+).*?vehicles:\s+([\d.]+)', text)
    if m:
        data['density_l1'] = float(m.group(1))
        data['avg_vehicles_l1'] = float(m.group(2))
    m = re.search(r'Lane 2:.*?density:\s+([\d.]+).*?vehicles:\s+([\d.]+)', text)
    if m:
        data['density_l2'] = float(m.group(1))
        data['avg_vehicles_l2'] = float(m.group(2))
    return data


def _parse_optimized_report_json():
    """Read the most recent optimized report JSON for full stats."""
    reports = sorted(REPORT_DIR.glob("optimized_*.json"))
    if not reports:
        return {}
    with open(reports[-1]) as f:
        return json.load(f)


# ── Pretty printer ──────────────────────────────────────────────

def _fmt(val, fmt_str=".2f", suffix=""):
    """Format a value, returning '—' if None/missing."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:{fmt_str}}{suffix}"
    return f"{val}{suffix}"


def _print_comparison(base, adv, opt):
    """Print a comprehensive 3-column comparison table."""
    W = 82
    HL = "═" * W

    print(f"\n{HL}")
    print(f"  ◆ BENCHMARK: 3-Way Pipeline Comparison")
    print(f"  ◆ {datetime.now().isoformat()}")
    print(HL)

    # Helper to print one row
    def row(label, b_val, a_val, o_val):
        print(f"  │ {label:<22} │ {b_val:>14} │ {a_val:>14} │ {o_val:>14} │")

    def header():
        print(f"  ┌────────────────────────┬────────────────┬────────────────┬────────────────┐")
        row("Metric", "Base", "Advanced", "Optimized")
        print(f"  ├────────────────────────┼────────────────┼────────────────┼────────────────┤")

    def footer():
        print(f"  └────────────────────────┴────────────────┴────────────────┴────────────────┘")

    def divider():
        print(f"  ├────────────────────────┼────────────────┼────────────────┼────────────────┤")

    # ── EXECUTION ──
    print(f"\n  {'EXECUTION':─<{W - 2}}")
    header()
    row("Time",
        _fmt(base.get('time_s'), ".2f", "s"),
        _fmt(adv.get('time_s'), ".2f", "s"),
        _fmt(opt.get('time_s'), ".2f", "s"))
    row("Memory",
        _fmt(base.get('memory_mb'), ".1f", " MB"),
        _fmt(adv.get('memory_mb'), ".1f", " MB"),
        _fmt(opt.get('memory_mb'), ".1f", " MB"))
    row("Frames",
        _fmt(base.get('frames'), "d"),
        _fmt(adv.get('frames'), "d"),
        _fmt(opt.get('frames'), "d"))
    row("FPS",
        _fmt(base.get('fps'), ".1f"),
        _fmt(adv.get('fps'), ".1f"),
        _fmt(opt.get('fps'), ".1f"))
    footer()

    # ── DENSITY ──
    print(f"\n  {'DENSITY':─<{W - 2}}")
    header()
    row("Lane 1 Avg",
        _fmt(base.get('density_l1'), ".4f"),
        _fmt(adv.get('density_l1'), ".4f"),
        _fmt(opt.get('density_l1'), ".4f"))
    row("Lane 2 Avg",
        _fmt(base.get('density_l2'), ".4f"),
        _fmt(adv.get('density_l2'), ".4f"),
        _fmt(opt.get('density_l2'), ".4f"))
    footer()

    # ── VEHICLE COUNTING ──
    print(f"\n  {'VEHICLE COUNTING':─<{W - 2}}")
    header()
    row("Lane 1",
        _fmt(base.get('avg_vehicles_l1'), ".1f", " avg"),
        _fmt(adv.get('count_l1'), "d"),
        _fmt(opt.get('count_l1'), "d"))
    row("Lane 2",
        _fmt(base.get('avg_vehicles_l2'), ".1f", " avg"),
        _fmt(adv.get('count_l2'), "d"),
        _fmt(opt.get('count_l2'), "d"))
    row("Combined / Total",
        "—",
        _fmt(adv.get('count_total'), "d"),
        _fmt(opt.get('count_total'), "d"))
    footer()

    # ── SPEED ESTIMATION ──
    print(f"\n  {'SPEED ESTIMATION':─<{W - 2}}")
    header()
    row("Lane 1 Avg",
        "—",
        _fmt(adv.get('speed_l1'), ".1f", " km/h"),
        _fmt(opt.get('speed_l1'), ".1f", " km/h"))
    row("Lane 2 Avg",
        "—",
        _fmt(adv.get('speed_l2'), ".1f", " km/h"),
        _fmt(opt.get('speed_l2'), ".1f", " km/h"))
    footer()

    # ── TRACKING ──
    print(f"\n  {'TRACKING':─<{W - 2}}")
    header()
    row("Total IDs Assigned",
        "—",
        _fmt(adv.get('track_ids'), "d"),
        _fmt(opt.get('track_ids'), "d"))
    footer()

    # ── OPTIMIZATION GAINS ──
    print(f"\n  {'OPTIMIZATION GAINS':─<{W - 2}}")

    adv_t = adv.get('time_s')
    opt_t = opt.get('time_s')
    base_t = base.get('time_s')

    print(f"  ┌────────────────────────┬─────────────────────────────────────────────────┐")
    if adv_t and opt_t and adv_t > 0:
        speedup_vs_adv = adv_t / opt_t
        savings_vs_adv = (1 - opt_t / adv_t) * 100
        print(f"  │ vs Advanced            │  {speedup_vs_adv:.2f}x faster  ({savings_vs_adv:+.1f}% time){'':>16}│")
    if base_t and opt_t and base_t > 0:
        speedup_vs_base = base_t / opt_t
        savings_vs_base = (1 - opt_t / base_t) * 100
        print(f"  │ vs Base                │  {speedup_vs_base:.2f}x faster  ({savings_vs_base:+.1f}% time){'':>16}│")

    skip_pct = opt.get('skip_rate_pct')
    if skip_pct is not None:
        print(f"  │ Frames Skipped         │  {skip_pct:.1f}% of all frames{'':>28}│")
    print(f"  └────────────────────────┴─────────────────────────────────────────────────┘")

    # ── OPTIMIZATION BREAKDOWN ──
    opt_stats = opt.get('optimization_stats')
    opt_pcts = opt.get('optimization_pcts')
    if opt_stats:
        print(f"\n  {'OPTIMIZATION BREAKDOWN (Optimized Pipeline)':─<{W - 2}}")
        print(f"  ┌──────────────────────┬──────────┬──────────┐")
        print(f"  │ Strategy             │  Frames  │  Rate    │")
        print(f"  ├──────────────────────┼──────────┼──────────┤")
        for key in ['skipped_temporal', 'skipped_pixel', 'skipped_sentinel', 'computed']:
            label = key.replace('skipped_', 'Skip: ').replace('computed', 'Full Compute')
            cnt = opt_stats.get(key, 0)
            pct = opt_pcts.get(key, 0) if opt_pcts else 0
            print(f"  │ {label:<20} │ {cnt:>8} │ {pct:>6.1f}%  │")
        print(f"  └──────────────────────┴──────────┴──────────┘")

    # ── OUTPUT FILES ──
    print(f"\n  {'OUTPUT FILES':─<{W - 2}}")
    print(f"    Base:      output/videos/base_pipeline_output.avi")
    print(f"    Advanced:  output/videos/output_advanced_cv.mp4")
    print(f"    Optimized: output/videos/output_optimized.mp4")

    print(f"\n{HL}\n")

    return {
        "base": base,
        "advanced": adv,
        "optimized": opt,
    }


# ── Main benchmark runner ──

def run_benchmark():
    """Run all 3 pipelines and print comprehensive comparison."""
    W = 82
    HL = "═" * W

    print(f"\n{HL}")
    print("  ◆ LLVD — FULL BENCHMARK (3 Pipelines)")
    print(f"  ◆ {datetime.now().isoformat()}")
    print(HL)

    real_stdout = sys.stdout
    base_data, adv_data, opt_data = {}, {}, {}

    # ── Phase 1: Base Pipeline ──
    print(f"\n  {'PHASE 1/3: BASE PIPELINE':─<{W - 2}}\n")
    tee = TeeCapture(real_stdout)
    sys.stdout = tee
    try:
        from src.base_pipeline import MultiprocessingTPL
        processor = MultiprocessingTPL()
        processor.run()
    except Exception as e:
        print(f"\n  [ERROR] Base pipeline failed: {e}")
    finally:
        sys.stdout = real_stdout
    base_data = _parse_base_output(tee.getvalue())
    base_data['frames'] = len(processor.density_lane1) if hasattr(processor, 'density_lane1') else 0

    # ── Phase 2: Advanced Pipeline ──
    print(f"\n  {'PHASE 2/3: ADVANCED PIPELINE':─<{W - 2}}\n")
    tee = TeeCapture(real_stdout)
    sys.stdout = tee
    try:
        from src.advanced_pipeline import main as advanced_main
        advanced_main()
    except Exception as e:
        print(f"\n  [ERROR] Advanced pipeline failed: {e}")
    finally:
        sys.stdout = real_stdout
    adv_data = _parse_advanced_output(tee.getvalue())

    # ── Phase 3: Optimized Pipeline ──
    print(f"\n  {'PHASE 3/3: OPTIMIZED PIPELINE':─<{W - 2}}\n")
    try:
        from src.grid_manager.optimized_pipeline import main as optimized_main
        optimized_main()
    except Exception as e:
        print(f"\n  [ERROR] Optimized pipeline failed: {e}")

    # Read the optimized report JSON (has complete data)
    opt_json = _parse_optimized_report_json()
    if opt_json:
        e = opt_json.get('execution', {})
        d = opt_json.get('density', {})
        c = opt_json.get('counting', {})
        s = opt_json.get('speed', {})
        t = opt_json.get('tracking', {})
        o = opt_json.get('optimization', {})

        opt_data = {
            'time_s': e.get('time_s'),
            'memory_mb': e.get('memory_mb'),
            'frames': e.get('frames_processed'),
            'fps': e.get('fps'),
            'density_l1': d.get('lane1_avg'),
            'density_l2': d.get('lane2_avg'),
            'count_l1': c.get('lane1_total'),
            'count_l2': c.get('lane2_total'),
            'count_total': c.get('combined'),
            'speed_l1': s.get('lane1_avg_kmh'),
            'speed_l2': s.get('lane2_avg_kmh'),
            'track_ids': t.get('total_ids_assigned'),
            'skip_rate_pct': o.get('skip_rate_pct') if o else None,
            'optimization_stats': o.get('stats') if o else None,
            'optimization_pcts': o.get('percentages') if o else None,
        }

    # ── Print Comparison ──
    comparison = _print_comparison(base_data, adv_data, opt_data)

    # ── Save Reports ──
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = REPORT_DIR / f"benchmark_{ts}.json"
    md_path = REPORT_DIR / f"benchmark_{ts}.md"

    with open(json_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "base": base_data,
            "advanced": adv_data,
            "optimized": {k: v for k, v in opt_data.items()
                         if k not in ('optimization_stats', 'optimization_pcts')},
        }, f, indent=2, default=str)

    # Markdown report
    with open(md_path, 'w') as f:
        f.write(_generate_benchmark_markdown(base_data, adv_data, opt_data))

    print(f"  Reports saved:")
    print(f"    JSON: {json_path}")
    print(f"    MD:   {md_path}")
    print()


def _generate_benchmark_markdown(base, adv, opt):
    """Generate a markdown benchmark comparison report."""
    lines = []
    lines.append("# LLVD Benchmark Report — 3-Way Comparison\n")
    lines.append(f"**Generated:** {datetime.now().isoformat()}\n")

    lines.append("## Execution\n")
    lines.append("| Metric | Base | Advanced | Optimized |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Time | {_fmt(base.get('time_s'), '.2f', 's')} | {_fmt(adv.get('time_s'), '.2f', 's')} | {_fmt(opt.get('time_s'), '.2f', 's')} |")
    lines.append(f"| Memory | {_fmt(base.get('memory_mb'), '.1f', ' MB')} | {_fmt(adv.get('memory_mb'), '.1f', ' MB')} | {_fmt(opt.get('memory_mb'), '.1f', ' MB')} |")
    lines.append(f"| Frames | {_fmt(base.get('frames'), 'd')} | {_fmt(adv.get('frames'), 'd')} | {_fmt(opt.get('frames'), 'd')} |")
    lines.append(f"| FPS | {_fmt(base.get('fps'), '.1f')} | {_fmt(adv.get('fps'), '.1f')} | {_fmt(opt.get('fps'), '.1f')} |")

    lines.append("\n## Density\n")
    lines.append("| Lane | Base | Advanced | Optimized |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Lane 1 Avg | {_fmt(base.get('density_l1'), '.4f')} | {_fmt(adv.get('density_l1'), '.4f')} | {_fmt(opt.get('density_l1'), '.4f')} |")
    lines.append(f"| Lane 2 Avg | {_fmt(base.get('density_l2'), '.4f')} | {_fmt(adv.get('density_l2'), '.4f')} | {_fmt(opt.get('density_l2'), '.4f')} |")

    lines.append("\n## Vehicle Counting\n")
    lines.append("| Lane | Base | Advanced | Optimized |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Lane 1 | {_fmt(base.get('avg_vehicles_l1'), '.1f', ' avg')} | {_fmt(adv.get('count_l1'), 'd')} | {_fmt(opt.get('count_l1'), 'd')} |")
    lines.append(f"| Lane 2 | {_fmt(base.get('avg_vehicles_l2'), '.1f', ' avg')} | {_fmt(adv.get('count_l2'), 'd')} | {_fmt(opt.get('count_l2'), 'd')} |")
    lines.append(f"| Combined | — | {_fmt(adv.get('count_total'), 'd')} | {_fmt(opt.get('count_total'), 'd')} |")

    lines.append("\n## Speed Estimation\n")
    lines.append("| Lane | Base | Advanced | Optimized |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Lane 1 | — | {_fmt(adv.get('speed_l1'), '.1f', ' km/h')} | {_fmt(opt.get('speed_l1'), '.1f', ' km/h')} |")
    lines.append(f"| Lane 2 | — | {_fmt(adv.get('speed_l2'), '.1f', ' km/h')} | {_fmt(opt.get('speed_l2'), '.1f', ' km/h')} |")

    adv_t = adv.get('time_s')
    opt_t = opt.get('time_s')
    base_t = base.get('time_s')

    lines.append("\n## Optimization Gains\n")
    if adv_t and opt_t and adv_t > 0:
        lines.append(f"- **vs Advanced:** {adv_t/opt_t:.2f}x faster ({(1-opt_t/adv_t)*100:+.1f}% time)")
    if base_t and opt_t and base_t > 0:
        lines.append(f"- **vs Base:** {base_t/opt_t:.2f}x faster ({(1-opt_t/base_t)*100:+.1f}% time)")
    if opt.get('skip_rate_pct') is not None:
        lines.append(f"- **Frames skipped:** {opt['skip_rate_pct']:.1f}%")

    lines.append("\n## Output Videos\n")
    lines.append("| Pipeline | File |")
    lines.append("|---|---|")
    lines.append("| Base | `output/videos/base_pipeline_output.avi` |")
    lines.append("| Advanced | `output/videos/output_advanced_cv.mp4` |")
    lines.append("| Optimized | `output/videos/output_optimized.mp4` |")
    lines.append("")

    return "\n".join(lines)


if __name__ == '__main__':
    run_benchmark()
