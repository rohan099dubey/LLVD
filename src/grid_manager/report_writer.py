"""
Report Writer — Unified report generator for all LLVD pipelines.

Generates:
  1. Beautiful formatted console output
  2. JSON report file (machine-readable)
  3. Markdown report file (human-readable)

All reports are saved to output/reports/ with timestamps.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path


REPORT_DIR = Path("output/reports")


class PipelineReport:
    """
    Collects metrics during pipeline execution and generates reports.

    Usage:
        report = PipelineReport(pipeline_name="advanced")
        report.set_config(...)
        report.set_execution(time=37.77, memory_mb=51.64, frames=1659, fps=43.92)
        report.set_density(lane1_avg=0.0269, lane2_avg=0.0421)
        report.set_counting(lane1=14, lane2=21)
        report.set_speed(...)
        report.set_scene(...)
        report.set_tracking(...)
        report.set_optimization(...)   # only for optimized pipeline
        report.set_profiling(...)

        report.print_console()
        report.save()
    """

    def __init__(self, pipeline_name="advanced"):
        self.pipeline_name = pipeline_name
        self.timestamp = datetime.now().isoformat()
        self.data = {
            "pipeline": pipeline_name,
            "timestamp": self.timestamp,
            "config": {},
            "execution": {},
            "density": {},
            "counting": {},
            "speed": {},
            "scene": {},
            "tracking": {},
            "optimization": None,
            "profiling": None,
        }

    # ── Setters ──

    def set_config(self, video_path="", color_channel="", grid_rows=0,
                   grid_cols=0, roi1=None, roi2=None, batch_size=0,
                   num_workers=0, **extra):
        self.data["config"] = {
            "video_path": str(video_path),
            "color_channel": color_channel,
            "grid": f"{grid_rows}x{grid_cols}",
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "roi1": roi1,
            "roi2": roi2,
            "batch_size": batch_size,
            "num_workers": num_workers,
            **extra,
        }

    def set_execution(self, time_s=0.0, memory_mb=0.0, frames=0, fps=0.0):
        self.data["execution"] = {
            "time_s": round(time_s, 2),
            "memory_mb": round(memory_mb, 2),
            "frames_processed": frames,
            "fps": round(fps, 2),
        }

    def set_density(self, lane1_avg=0.0, lane2_avg=0.0,
                    lane1_values=None, lane2_values=None):
        self.data["density"] = {
            "lane1_avg": round(lane1_avg, 4),
            "lane2_avg": round(lane2_avg, 4),
        }
        if lane1_values is not None:
            self.data["density"]["lane1_max"] = round(max(lane1_values), 4) if lane1_values else 0.0
            self.data["density"]["lane2_max"] = round(max(lane2_values), 4) if lane2_values else 0.0
            self.data["density"]["lane1_min"] = round(min(lane1_values), 4) if lane1_values else 0.0
            self.data["density"]["lane2_min"] = round(min(lane2_values), 4) if lane2_values else 0.0

    def set_counting(self, lane1=0, lane2=0):
        self.data["counting"] = {
            "lane1_total": lane1,
            "lane2_total": lane2,
            "combined": lane1 + lane2,
        }

    def set_speed(self, lane1_avg=0.0, lane2_avg=0.0,
                  lane1_samples=0, lane2_samples=0):
        self.data["speed"] = {
            "lane1_avg_kmh": round(lane1_avg, 1),
            "lane2_avg_kmh": round(lane2_avg, 1),
            "lane1_samples": lane1_samples,
            "lane2_samples": lane2_samples,
        }

    def set_scene(self, scene_counts=None, initial_lighting="",
                  initial_foggy=False, brightness=0.0, fog_index=0.0):
        self.data["scene"] = {
            "initial_lighting": initial_lighting,
            "initial_foggy": initial_foggy,
            "brightness": round(brightness, 1),
            "fog_index": round(fog_index, 2),
            "conditions": dict(scene_counts) if scene_counts else {},
        }

    def set_tracking(self, total_ids=0, active_l1=0, active_l2=0):
        self.data["tracking"] = {
            "total_ids_assigned": total_ids,
            "active_l1_tracks": active_l1,
            "active_l2_tracks": active_l2,
        }

    def set_optimization(self, stats=None, config=None):
        """Set optimization stats — only used by the optimized pipeline."""
        if stats is None:
            return
        total = sum(stats.values())
        pcts = {k: round(v / total * 100, 1) if total else 0 for k, v in stats.items()}
        skip_total = total - stats.get('computed', 0)
        self.data["optimization"] = {
            "enabled": True,
            "config": config or {},
            "stats": stats,
            "percentages": pcts,
            "total_frames": total,
            "total_skipped": skip_total,
            "skip_rate_pct": round(skip_total / total * 100, 1) if total else 0,
        }

    def set_profiling(self, profiling_stats=None):
        """Set function-level profiling data."""
        if not profiling_stats:
            return
        formatted = {}
        for name, s in profiling_stats.items():
            calls = s.get('call_count', 0)
            total = s.get('total_time', 0)
            if calls > 0:
                formatted[name] = {
                    "calls": calls,
                    "total_s": round(total, 4),
                    "avg_ms": round((total / calls) * 1000, 4),
                    "min_ms": round((s.get('min_time', 0) or 0) * 1000, 4),
                    "max_ms": round(s.get('max_time', 0) * 1000, 4),
                }
        # Sort by total time descending
        self.data["profiling"] = dict(
            sorted(formatted.items(), key=lambda x: x[1]['total_s'], reverse=True)
        )

    # ── Console Output ──

    def print_console(self):
        """Print a formatted report to the console."""
        e = self.data["execution"]
        d = self.data["density"]
        c = self.data["counting"]
        sp = self.data["speed"]
        sc = self.data["scene"]
        t = self.data["tracking"]
        cfg = self.data["config"]
        opt = self.data.get("optimization")
        prof = self.data.get("profiling")

        W = 78  # console width
        HL = "═" * W
        SL = "─" * W

        pipeline_label = {
            "advanced": "ADVANCED CLASSICAL CV",
            "optimized": "OPTIMIZED GRID MANAGER",
            "base": "BASE PIPELINE (DBSCAN)",
        }.get(self.pipeline_name, self.pipeline_name.upper())

        # Header
        print(f"\n{HL}")
        print(f"  ◆ LLVD — {pipeline_label}")
        print(f"  ◆ {self.timestamp}")
        print(HL)

        # Config
        print(f"\n  {'CONFIG':─<{W - 2}}")
        print(f"  Video:       {cfg.get('video_path', '?')}")
        print(f"  Grid:        {cfg.get('grid', '?')} | Channel: {cfg.get('color_channel', '?')}")
        print(f"  Workers:     {cfg.get('num_workers', '?')} | Batch: {cfg.get('batch_size', '?')}")

        # Execution
        print(f"\n  {'EXECUTION':─<{W - 2}}")
        print(f"  ┌─────────────────────┬─────────────────────┐")
        print(f"  │ Time:   {e.get('time_s', 0):>10.2f}s  │ Memory: {e.get('memory_mb', 0):>9.2f} MB │")
        print(f"  │ Frames: {e.get('frames_processed', 0):>10}   │ Speed:  {e.get('fps', 0):>9.2f} FPS│")
        print(f"  └─────────────────────┴─────────────────────┘")

        # Density
        if d:
            print(f"\n  {'DENSITY':─<{W - 2}}")
            print(f"  ┌─────────────────────┬─────────────────────┐")
            print(f"  │ Lane 1 Avg: {d.get('lane1_avg', 0):>8.4f} │ Lane 2 Avg: {d.get('lane2_avg', 0):>8.4f}│")
            if 'lane1_max' in d:
                print(f"  │ Lane 1 Max: {d.get('lane1_max', 0):>8.4f} │ Lane 2 Max: {d.get('lane2_max', 0):>8.4f}│")
                print(f"  │ Lane 1 Min: {d.get('lane1_min', 0):>8.4f} │ Lane 2 Min: {d.get('lane2_min', 0):>8.4f}│")
            print(f"  └─────────────────────┴─────────────────────┘")

        # Counting
        if c:
            print(f"\n  {'VEHICLE COUNTING':─<{W - 2}}")
            print(f"  ┌─────────────────────┬─────────────────────┐")
            print(f"  │ Lane 1:     {c.get('lane1_total', 0):>7}   │ Lane 2:     {c.get('lane2_total', 0):>7}  │")
            print(f"  │ Combined:   {c.get('combined', 0):>7}   │                      │")
            print(f"  └─────────────────────┴─────────────────────┘")

        # Speed
        if sp and sp.get("lane1_samples", 0) > 0:
            print(f"\n  {'SPEED ESTIMATION':─<{W - 2}}")
            print(f"  ┌─────────────────────┬─────────────────────┐")
            print(f"  │ Lane 1: {sp.get('lane1_avg_kmh', 0):>7.1f} km/h │ Lane 2: {sp.get('lane2_avg_kmh', 0):>7.1f} km/h│")
            print(f"  │ Samples:    {sp.get('lane1_samples', 0):>7}   │ Samples:    {sp.get('lane2_samples', 0):>7}  │")
            print(f"  └─────────────────────┴─────────────────────┘")

        # Scene
        if sc and sc.get("conditions"):
            print(f"\n  {'SCENE CONDITIONS':─<{W - 2}}")
            for cond, cnt in sorted(sc["conditions"].items()):
                print(f"    {cond:<20} {cnt} batches")

        # Tracking
        if t and t.get("total_ids_assigned", 0) > 0:
            print(f"\n  {'TRACKING':─<{W - 2}}")
            print(f"  ┌─────────────────────────────────────────────┐")
            print(f"  │ Total IDs Assigned: {t.get('total_ids_assigned', 0):>7}                  │")
            print(f"  │ Active L1 Tracks:   {t.get('active_l1_tracks', 0):>7}                  │")
            print(f"  │ Active L2 Tracks:   {t.get('active_l2_tracks', 0):>7}                  │")
            print(f"  └─────────────────────────────────────────────┘")

        # Optimization (only for optimized pipeline)
        if opt and opt.get("enabled"):
            print(f"\n  {'OPTIMIZATION STATS':─<{W - 2}}")
            stats = opt.get("stats", {})
            pcts = opt.get("percentages", {})
            print(f"  ┌──────────────────────┬──────────┬──────────┐")
            print(f"  │ Strategy             │  Frames  │  Rate    │")
            print(f"  ├──────────────────────┼──────────┼──────────┤")
            for key in ['skipped_temporal', 'skipped_pixel', 'skipped_sentinel', 'computed']:
                label = key.replace('skipped_', 'Skip: ').replace('computed', 'Full Compute')
                cnt = stats.get(key, 0)
                pct = pcts.get(key, 0)
                print(f"  │ {label:<20} │ {cnt:>8} │ {pct:>6.1f}%  │")
            print(f"  ├──────────────────────┼──────────┼──────────┤")
            print(f"  │ {'Total Skipped':<20} │ {opt.get('total_skipped', 0):>8} │ {opt.get('skip_rate_pct', 0):>6.1f}%  │")
            print(f"  └──────────────────────┴──────────┴──────────┘")

        # Profiling
        if prof:
            print(f"\n  {'FUNCTION PROFILING':─<{W - 2}}")
            print(f"  {'Function':<32} {'Calls':<8} {'Total(s)':<10} {'Avg(ms)':<10} {'Max(ms)':<10}")
            print(f"  {SL}")
            for name, s in list(prof.items())[:15]:  # top 15
                print(f"  {name:<32} {s['calls']:<8} {s['total_s']:<10.4f} {s['avg_ms']:<10.4f} {s['max_ms']:<10.4f}")

        print(f"\n{HL}\n")

    # ── File Output ──

    def save(self):
        """Save reports to output/reports/ as both JSON and Markdown."""
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.pipeline_name}_{ts}"

        json_path = REPORT_DIR / f"{base_name}.json"
        md_path = REPORT_DIR / f"{base_name}.md"

        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

        # Save Markdown
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown())

        print(f"  Reports saved:")
        print(f"    JSON: {json_path}")
        print(f"    MD:   {md_path}")

        return json_path, md_path

    def _generate_markdown(self):
        """Generate a Markdown report."""
        e = self.data["execution"]
        d = self.data["density"]
        c = self.data["counting"]
        sp = self.data["speed"]
        sc = self.data["scene"]
        t = self.data["tracking"]
        cfg = self.data["config"]
        opt = self.data.get("optimization")
        prof = self.data.get("profiling")

        pipeline_label = {
            "advanced": "Advanced Classical CV",
            "optimized": "Optimized Grid Manager",
            "base": "Base Pipeline (DBSCAN)",
        }.get(self.pipeline_name, self.pipeline_name)

        lines = []
        lines.append(f"# LLVD Pipeline Report — {pipeline_label}")
        lines.append(f"\n**Generated:** {self.timestamp}  ")
        lines.append(f"**Video:** `{cfg.get('video_path', '?')}`  ")
        lines.append(f"**Grid:** {cfg.get('grid', '?')} | **Channel:** {cfg.get('color_channel', '?')}  ")
        lines.append(f"**Workers:** {cfg.get('num_workers', '?')} | **Batch:** {cfg.get('batch_size', '?')}")

        # Execution
        lines.append(f"\n## Execution Summary\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Time | {e.get('time_s', 0):.2f}s |")
        lines.append(f"| Memory | {e.get('memory_mb', 0):.2f} MB |")
        lines.append(f"| Frames Processed | {e.get('frames_processed', 0)} |")
        lines.append(f"| Processing Speed | {e.get('fps', 0):.2f} FPS |")

        # Density
        if d:
            lines.append(f"\n## Density Analysis\n")
            lines.append(f"| Lane | Avg | Min | Max |")
            lines.append(f"|---|---|---|---|")
            lines.append(f"| Lane 1 | {d.get('lane1_avg', 0):.4f} | {d.get('lane1_min', '—')} | {d.get('lane1_max', '—')} |")
            lines.append(f"| Lane 2 | {d.get('lane2_avg', 0):.4f} | {d.get('lane2_min', '—')} | {d.get('lane2_max', '—')} |")

        # Counting
        if c:
            lines.append(f"\n## Vehicle Counting\n")
            lines.append(f"| Lane | Count |")
            lines.append(f"|---|---|")
            lines.append(f"| Lane 1 | {c.get('lane1_total', 0)} |")
            lines.append(f"| Lane 2 | {c.get('lane2_total', 0)} |")
            lines.append(f"| **Combined** | **{c.get('combined', 0)}** |")

        # Speed
        if sp and sp.get("lane1_samples", 0) > 0:
            lines.append(f"\n## Speed Estimation\n")
            lines.append(f"| Lane | Avg Speed | Samples |")
            lines.append(f"|---|---|---|")
            lines.append(f"| Lane 1 | {sp.get('lane1_avg_kmh', 0):.1f} km/h | {sp.get('lane1_samples', 0)} |")
            lines.append(f"| Lane 2 | {sp.get('lane2_avg_kmh', 0):.1f} km/h | {sp.get('lane2_samples', 0)} |")

        # Scene
        if sc and sc.get("conditions"):
            lines.append(f"\n## Scene Conditions\n")
            lines.append(f"**Initial:** {sc.get('initial_lighting', '?').upper()}"
                         f"{' + FOG' if sc.get('initial_foggy') else ''}"
                         f" (brightness={sc.get('brightness', 0):.1f}, fog_idx={sc.get('fog_index', 0):.2f})\n")
            lines.append(f"| Condition | Batches |")
            lines.append(f"|---|---|")
            for cond, cnt in sorted(sc["conditions"].items()):
                lines.append(f"| {cond} | {cnt} |")

        # Tracking
        if t and t.get("total_ids_assigned", 0) > 0:
            lines.append(f"\n## Tracking\n")
            lines.append(f"| Metric | Value |")
            lines.append(f"|---|---|")
            lines.append(f"| Total IDs Assigned | {t.get('total_ids_assigned', 0)} |")
            lines.append(f"| Active L1 Tracks | {t.get('active_l1_tracks', 0)} |")
            lines.append(f"| Active L2 Tracks | {t.get('active_l2_tracks', 0)} |")

        # Optimization
        if opt and opt.get("enabled"):
            lines.append(f"\n## Optimization Stats\n")
            stats = opt.get("stats", {})
            pcts = opt.get("percentages", {})
            lines.append(f"| Strategy | Frames | Rate |")
            lines.append(f"|---|---|---|")
            for key in ['skipped_temporal', 'skipped_pixel', 'skipped_sentinel', 'computed']:
                label = key.replace('skipped_', 'Skip: ').replace('computed', 'Full Compute')
                lines.append(f"| {label} | {stats.get(key, 0)} | {pcts.get(key, 0):.1f}% |")
            lines.append(f"| **Total Skipped** | **{opt.get('total_skipped', 0)}** | **{opt.get('skip_rate_pct', 0):.1f}%** |")

        # Profiling
        if prof:
            lines.append(f"\n## Function Profiling\n")
            lines.append(f"| Function | Calls | Total (s) | Avg (ms) | Max (ms) |")
            lines.append(f"|---|---|---|---|---|")
            for name, s in list(prof.items())[:15]:
                lines.append(f"| `{name}` | {s['calls']} | {s['total_s']:.4f} | {s['avg_ms']:.4f} | {s['max_ms']:.4f} |")

        lines.append("")
        return "\n".join(lines)
