"""Exit 0 if attn_share GPU-hours spent + this run's estimate + reserve stay within the 5.0 cap."""
import csv, sys
spent = sum(float(r["gpu_hours"]) for r in csv.DictReader(open("results/gpu_hours.csv")) if r["stage"] == "attn_share")
est, reserve = float(sys.argv[1]), float(sys.argv[2])
ok = spent + est + reserve <= 5.0
print(f"budget: spent {spent:.2f} + est {est} + reserve {reserve} -> {'ok' if ok else 'SKIP (cap)'}", flush=True)
sys.exit(0 if ok else 1)
