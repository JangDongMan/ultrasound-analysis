"""
Auto-mark VB5K-converted files in data/ directory.

VB5K signals are rectified (all-positive), so raw ADC values are used
as the envelope directly. Cluster detection uses a moving-average
smoothed envelope with an adaptive threshold.

Cluster grouping: clusters separated by < GROUP_GAP_US are merged
into one group. The first group → dermis region, second → fascia.

JSON output format matches usdata boundary_marker_gui format.

Usage:
    python auto_mark_data.py            # dry-run
    python auto_mark_data.py --apply    # write JSON files
    python auto_mark_data.py --overwrite  # overwrite existing JSON
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.ndimage import uniform_filter1d

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VB5K_START_US = 18.50
SAMPLE_STEP_US = 0.01
SPEED_OF_SOUND = 1540.0

SMOOTH_SAMPLES = 30         # moving-average window (0.3μs)
MIN_CLUSTER_SAMPLES = 20    # minimum cluster width (0.2μs)
THRESHOLD_RATIO = 0.15      # adaptive: 15% of signal max
THRESHOLD_MIN = 20          # absolute floor threshold (ADC units)
GROUP_GAP_US = 1.5          # clusters within 1.5μs are merged into same group


def detect_clusters(arr):
    """Detect clusters in VB5K rectified signal. Returns list of (start, end) tuples."""
    env = uniform_filter1d(arr.astype(float), size=SMOOTH_SAMPLES)
    threshold = max(THRESHOLD_MIN, env.max() * THRESHOLD_RATIO)

    above = env > threshold
    clusters = []
    in_c, cs = False, 0
    for i, v in enumerate(above):
        if v and not in_c:
            in_c, cs = True, i
        elif not v and in_c:
            in_c = False
            if i - 1 - cs >= MIN_CLUSTER_SAMPLES:
                clusters.append((cs, i - 1))
    if in_c and len(env) - 1 - cs >= MIN_CLUSTER_SAMPLES:
        clusters.append((cs, len(env) - 1))

    return clusters


def group_clusters(clusters, gap_us=GROUP_GAP_US):
    """Merge clusters separated by < gap_us into groups.

    Returns list of (group_start_sample, group_end_sample).
    """
    if not clusters:
        return []

    gap_samples = int(gap_us / SAMPLE_STEP_US)
    groups = [(clusters[0][0], clusters[0][1])]

    for cs, ce in clusters[1:]:
        gs, ge = groups[-1]
        if cs - ge < gap_samples:
            groups[-1] = (gs, ce)
        else:
            groups.append((cs, ce))

    return groups


def sample_to_us(sample_idx):
    return VB5K_START_US + sample_idx * SAMPLE_STEP_US


def auto_mark(csv_path):
    """Auto-mark one file. Returns JSON dict or None if detection fails."""
    arr = np.loadtxt(csv_path, delimiter=",").flatten().astype(float)
    if len(arr) != 2050:
        return None  # not yet converted

    clusters = detect_clusters(arr)
    if len(clusters) < 2:
        return None

    groups = group_clusters(clusters)
    if len(groups) < 2:
        return None

    g1_start, g1_end = groups[0]
    g2_start, g2_end = groups[1]

    start_us = sample_to_us(g1_start)
    pos1_us = sample_to_us(g1_end)    # first group end = dermis boundary
    pos2_us = sample_to_us(g2_start)  # second group start = fascia

    thickness1_mm = (pos1_us - start_us) * SPEED_OF_SOUND / 2000.0
    depth2_mm = (pos2_us - start_us) * SPEED_OF_SOUND / 2000.0

    return {
        "source_file": os.path.basename(csv_path),
        "start_point_us": round(start_us, 4),
        "num_positions": 2,
        "speed_of_sound": SPEED_OF_SOUND,
        "sample_interval_ns": 10,
        "num_samples": 2050,
        "auto_marked": True,
        "positions": [
            {
                "position_number": 1,
                "position_name": "피하지방시작",
                "time_us": round(pos1_us, 4),
                "thickness_mm": round(thickness1_mm, 4),
                "depth_start_mm": 0.0,
                "depth_end_mm": round(thickness1_mm, 4),
            },
            {
                "position_number": 2,
                "position_name": "Fascia",
                "time_us": round(pos2_us, 4),
                "thickness_mm": round(depth2_mm - thickness1_mm, 4),
                "depth_start_mm": round(thickness1_mm, 4),
                "depth_end_mm": round(depth2_mm, 4),
            },
        ],
    }


def json_path_for(csv_path):
    return os.path.splitext(csv_path)[0] + "_positions.json"


def run(apply=False, overwrite=False):
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    ok = skipped = failed = 0

    for csv_path in csv_files:
        jp = json_path_for(csv_path)
        if os.path.exists(jp) and not overwrite:
            skipped += 1
            continue

        result = auto_mark(csv_path)
        if result is None:
            print(f"  [FAIL] {os.path.basename(csv_path)}")
            failed += 1
            continue

        p1 = result["positions"][0]
        p2 = result["positions"][1]
        print(
            f"  [OK]   {os.path.basename(csv_path)}"
            f"  t0={result['start_point_us']:.2f}μs"
            f"  dermis={p1['time_us']:.2f}μs({p1['thickness_mm']:.2f}mm)"
            f"  fascia={p2['time_us']:.2f}μs({p2['depth_end_mm']:.2f}mm)"
        )

        if apply:
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        ok += 1

    print()
    action = "marked" if apply else "would mark"
    print(f"Result: {ok} {action}, {skipped} skipped (JSON exists), {failed} failed")
    if not apply and ok > 0:
        print("Run with --apply to write JSON files.")


def main():
    parser = argparse.ArgumentParser(description="Auto-mark VB5K files in data/")
    parser.add_argument("--apply", action="store_true", help="Write JSON files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON")
    args = parser.parse_args()
    run(apply=args.apply, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
