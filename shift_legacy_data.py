"""
Convert legacy CSV+JSON files to new format (drop=1850, 2050 samples).

Two source formats supported:
  [A] usdata/ integer ADC files (1250 samples, drop=1200):
      CSV: prepend 650×128, keep 1250, append 150×128 → 2050
      JSON: all time_us += 6.5μs

  [B] data/ VB5K oscilloscope files (time,voltage columns, ~2000 samples):
      CSV: voltage→ADC, align to 18.5μs start, make 2050 samples
      JSON: time_us already absolute, copy as-is (NO shift)

Usage:
    python shift_legacy_data.py            # dry-run (no changes)
    python shift_legacy_data.py --apply    # execute conversion
    python shift_legacy_data.py --verify   # validate already-converted files
"""

import argparse
import glob
import json
import os

import numpy as np

SHIFT_SAMPLES = 650        # usdata shift: NEW_DROP - OLD_DROP = 1850 - 1200
SHIFT_US = 6.5             # 650 * 10ns
OLD_TRIM = 1250
NEW_TRIM = 2050
PAD_TAIL = NEW_TRIM - OLD_TRIM - SHIFT_SAMPLES  # 150

VB5K_START_US = 18.50      # new format reference start
SAMPLE_STEP_US = 0.01      # 10ns per sample
VREF = 3.0                 # VB5K voltage reference (V) — signals up to ~2.9V, must not clip

DATA_DIRS = [
    os.path.join(os.path.dirname(__file__), "usdata"),
    os.path.join(os.path.dirname(__file__), "data"),
]


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------

def _is_vb5k(path):
    """Detect VB5K oscilloscope format by header lines."""
    try:
        with open(path, "r") as f:
            for _ in range(5):
                line = f.readline()
                if "x-axis" in line or "second,Volt" in line:
                    return True
    except Exception:
        pass
    return False


def classify_file(path):
    """Return ('vb5k'|'old_int'|'new_int'|'skip'|'error', arr_or_None)."""
    if _is_vb5k(path):
        return "vb5k", None
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32).flatten()
        n = len(arr)
        if n == OLD_TRIM:
            return "old_int", arr
        elif n == NEW_TRIM:
            return "new_int", arr
        else:
            return "skip", arr
    except Exception:
        return "error", None


def find_csv_files():
    found = []
    for base in DATA_DIRS:
        if not os.path.isdir(base):
            continue
        for path in glob.glob(os.path.join(base, "**", "*.csv"), recursive=True):
            found.append(path)
    return sorted(found)


def json_path_for(csv_path):
    return os.path.splitext(csv_path)[0] + "_positions.json"


# ---------------------------------------------------------------------------
# usdata integer ADC conversion (type A)
# ---------------------------------------------------------------------------

def shift_csv_int(arr):
    """1250 → 2050: prepend 650×128, append 150×128."""
    return np.concatenate([
        np.full(SHIFT_SAMPLES, 128, dtype=arr.dtype),
        arr,
        np.full(PAD_TAIL, 128, dtype=arr.dtype),
    ])


def shift_json(data):
    """Return new dict with all time_us += 6.5μs."""
    nd = dict(data)
    if "start_point_us" in nd:
        nd["start_point_us"] = round(nd["start_point_us"] + SHIFT_US, 4)
    if "positions" in nd:
        nd["positions"] = [
            {**p, "time_us": round(p["time_us"] + SHIFT_US, 4)}
            if "time_us" in p else dict(p)
            for p in nd["positions"]
        ]
    return nd


# ---------------------------------------------------------------------------
# VB5K oscilloscope conversion (type B)
# ---------------------------------------------------------------------------

def load_vb5k(path):
    """Load VB5K file → (time_us array, adc_int array)."""
    t_list, adc_list = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "x-axis" in line or "second" in line:
                continue
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip():
                try:
                    t_us = float(parts[0]) * 1e6
                    v = float(parts[1])
                    adc = int(np.clip(v / VREF * 255, 0, 255))
                    t_list.append(t_us)
                    adc_list.append(adc)
                except ValueError:
                    continue
    return np.array(t_list), np.array(adc_list, dtype=np.int32)


def convert_vb5k_to_new(path):
    """Convert VB5K file to 2050-sample integer array aligned at 18.5μs."""
    t_us, adc = load_vb5k(path)
    if len(t_us) == 0:
        raise ValueError("No valid data in VB5K file")

    new_adc = np.full(NEW_TRIM, 128, dtype=np.int32)
    for i in range(NEW_TRIM):
        target = VB5K_START_US + i * SAMPLE_STEP_US
        idx = np.searchsorted(t_us, target)
        if idx < len(t_us):
            new_adc[i] = adc[idx]
    return new_adc


def update_vb5k_json(data):
    """Update VB5K JSON metadata (num_samples). time_us already absolute → no shift."""
    nd = dict(data)
    nd["num_samples"] = NEW_TRIM
    return nd


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run():
    csv_files = find_csv_files()
    counts = {"old_int": 0, "new_int": 0, "vb5k": 0, "skip": 0, "error": 0}
    json_missing = {"old_int": 0, "vb5k": 0}

    for csv_path in csv_files:
        kind, arr = classify_file(csv_path)
        counts[kind] = counts.get(kind, 0) + 1
        jp = json_path_for(csv_path)
        has_json = os.path.exists(jp)

        if kind == "old_int":
            if not has_json:
                json_missing["old_int"] += 1
            print(f"  [OLD_INT] {os.path.relpath(csv_path)} (json={'yes' if has_json else 'MISSING'})")
        elif kind == "vb5k":
            if not has_json:
                json_missing["vb5k"] += 1
            print(f"  [VB5K]    {os.path.relpath(csv_path)} (json={'yes' if has_json else 'MISSING'})")
        elif kind == "skip":
            print(f"  [SKIP]    {os.path.relpath(csv_path)} ({len(arr)} samples)")
        elif kind == "error":
            print(f"  [ERR]     {os.path.relpath(csv_path)}")

    print()
    print("Summary (dry-run):")
    print(f"  [A] Integer ADC 1250-sample (will shift +650): {counts['old_int']}")
    print(f"        JSON missing: {json_missing['old_int']}")
    print(f"  [B] VB5K oscilloscope (will convert+align):    {counts['vb5k']}")
    print(f"        JSON missing: {json_missing['vb5k']}")
    print(f"  Already new (2050-sample):                     {counts['new_int']}")
    print(f"  Skip (other):                                  {counts.get('skip',0)}")
    print(f"  Error:                                         {counts.get('error',0)}")
    total = counts["old_int"] + counts["vb5k"]
    if total > 0:
        print(f"\nRun with --apply to convert {total} files.")


# ---------------------------------------------------------------------------
# Apply conversion
# ---------------------------------------------------------------------------

def apply_conversion():
    csv_files = find_csv_files()
    converted = skipped = errors = 0

    for csv_path in csv_files:
        kind, arr = classify_file(csv_path)

        if kind == "old_int":
            try:
                new_arr = shift_csv_int(arr)
                np.savetxt(csv_path, new_arr.reshape(-1, 1), fmt="%d", delimiter=",")
                jp = json_path_for(csv_path)
                if os.path.exists(jp):
                    with open(jp, "r", encoding="utf-8") as f:
                        jdata = json.load(f)
                    with open(jp, "w", encoding="utf-8") as f:
                        json.dump(shift_json(jdata), f, ensure_ascii=False, indent=2)
                print(f"  [OK-INT]  {os.path.relpath(csv_path)}")
                converted += 1
            except Exception as e:
                print(f"  [ERR]     {os.path.relpath(csv_path)}: {e}")
                errors += 1

        elif kind == "vb5k":
            try:
                new_arr = convert_vb5k_to_new(csv_path)
                np.savetxt(csv_path, new_arr.reshape(-1, 1), fmt="%d", delimiter=",")
                jp = json_path_for(csv_path)
                if os.path.exists(jp):
                    with open(jp, "r", encoding="utf-8") as f:
                        jdata = json.load(f)
                    with open(jp, "w", encoding="utf-8") as f:
                        json.dump(update_vb5k_json(jdata), f, ensure_ascii=False, indent=2)
                print(f"  [OK-VB5K] {os.path.relpath(csv_path)}")
                converted += 1
            except Exception as e:
                print(f"  [ERR]     {os.path.relpath(csv_path)}: {e}")
                errors += 1

        else:
            skipped += 1

    print()
    print(f"Conversion complete: {converted} converted, {skipped} skipped, {errors} errors.")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify():
    csv_files = find_csv_files()
    ok = bad = 0

    for csv_path in csv_files:
        kind, arr = classify_file(csv_path)
        if kind != "new_int":
            continue

        issues = []
        front_mean = arr[:SHIFT_SAMPLES].mean()
        if abs(front_mean - 128) > 10:
            issues.append(f"front pad mean={front_mean:.1f}")

        jp = json_path_for(csv_path)
        if os.path.exists(jp):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    jdata = json.load(f)
                sp = jdata.get("start_point_us", 0)
                if sp < 18.0:
                    issues.append(f"start_point_us={sp:.3f} < 18.0")
                for pos in jdata.get("positions", []):
                    t = pos.get("time_us", 0)
                    if t < 18.0:
                        issues.append(f"time_us={t:.3f} < 18.0")
            except Exception as e:
                issues.append(f"JSON error: {e}")

        if issues:
            print(f"  [WARN] {os.path.relpath(csv_path)}: {'; '.join(issues)}")
            bad += 1
        else:
            ok += 1

    print()
    print(f"Verify: {ok} OK, {bad} with issues.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert legacy ultrasound CSV+JSON to new 2050-sample format"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="Execute conversion")
    group.add_argument("--verify", action="store_true", help="Validate converted files")
    args = parser.parse_args()

    if args.apply:
        print("=== APPLYING CONVERSION ===")
        apply_conversion()
    elif args.verify:
        print("=== VERIFYING CONVERTED FILES ===")
        verify()
    else:
        print("=== DRY-RUN (no changes) ===")
        dry_run()


if __name__ == "__main__":
    main()
