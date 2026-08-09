import argparse
import csv
import os
import statistics
from collections import defaultdict


def _f(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _fmt(values, unit="", decimals=2):
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    mean = statistics.mean(vals)
    if len(vals) < 2:
        return f"{mean:.{decimals}f}{unit} (n=1)"
    sd = statistics.stdev(vals)
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}{unit} (n={len(vals)})"


def _median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="logs/baseline_fault_results.csv")
    parser.add_argument("--out", type=str, default="logs/baseline_summary.csv",
                        help="Where to write the machine-readable summary")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found.")
        print("Run baseline_fault_eval.py first, or pass --csv with the right path.")
        return

    by_fault = defaultdict(list)
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            by_fault[row["fault_type"]].append(row)

    if not by_fault:
        print(f"ERROR: {args.csv} has no data rows.")
        return

    total = sum(len(rows) for rows in by_fault.values())
    print("=" * 78)
    print(f"BASELINE A (no adaptation) -- {total} trials across {len(by_fault)} fault types")
    print(f"source: {args.csv}")
    print("=" * 78)

    summary_rows = []

    for fault_type in sorted(by_fault):
        rows = by_fault[fault_type]
        n = len(rows)

        recovery_times = [_f(r["recovery_time_s"]) for r in rows]
        recovered = [t for t in recovery_times if t is not None]
        distances = [_f(r["post_fault_distance_m"]) for r in rows]
        survived = [_f(r["post_fault_steps_survived"]) for r in rows]
        baselines = [_f(r["baseline_vel"]) for r in rows]
        falls = [str(r["fell"]).strip().lower() in ("true", "1", "yes") for r in rows]

        recovery_rate = len(recovered) / n if n else 0.0
        fall_rate = sum(falls) / n if n else 0.0
        med = _median(recovered)

        print(f"\n--- {fault_type}  (severity {rows[0]['severity']}, n={n}) ---")
        print(f"  Recovery rate      : {recovery_rate:6.1%}   "
              f"({len(recovered)}/{n} regained pre-fault speed)")
        if recovered:
            print(f"  Recovery time      : {_fmt(recovered, ' s')}"
                  + (f"   median {med:.2f} s" if med is not None else ""))
        else:
            print(f"  Recovery time      : never recovered in any trial")
        print(f"  Fall rate          : {fall_rate:6.1%}   ({sum(falls)}/{n} fell)")
        print(f"  Post-fault distance: {_fmt(distances, ' m')}")
        print(f"  Steps survived     : {_fmt(survived, '', 1)}")
        print(f"  Pre-fault speed    : {_fmt(baselines, ' m/s')}")

        summary_rows.append({
            "fault_type": fault_type,
            "severity": rows[0]["severity"],
            "n_trials": n,
            "recovery_rate": round(recovery_rate, 4),
            "n_recovered": len(recovered),
            "recovery_time_mean_s": round(statistics.mean(recovered), 4) if recovered else "",
            "recovery_time_median_s": round(med, 4) if med is not None else "",
            "fall_rate": round(fall_rate, 4),
            "post_fault_distance_mean_m": round(statistics.mean([d for d in distances if d is not None]), 4)
                if any(d is not None for d in distances) else "",
            "steps_survived_mean": round(statistics.mean([s for s in survived if s is not None]), 2)
                if any(s is not None for s in survived) else "",
            "baseline_vel_mean": round(statistics.mean([b for b in baselines if b is not None]), 4)
                if any(b is not None for b in baselines) else "",
        })

    # ranking which faults hurt most
    print("\n" + "=" * 78)
    print("SEVERITY RANKING (hardest fault first, by recovery rate)")
    print("=" * 78)
    ranked = sorted(summary_rows, key=lambda r: (r["recovery_rate"], -r["fall_rate"]))
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['fault_type']:<18} recovery {r['recovery_rate']:6.1%}   "
              f"falls {r['fall_rate']:6.1%}")

    print("\nThis ranking is the headline Baseline A result: it says which failure")
    print("modes an unadapted policy genuinely cannot handle. Those are the ones")
    print("where an adaptation method has room to show a real improvement.")

    # write machine-readable summary
    if summary_rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nWrote summary table to {args.out}")


if __name__ == "__main__":
    main()
