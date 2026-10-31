"""Common budget for the 2026-10-31 handoff "audit and replicate anchor-aware calibration": sums tags bar1 (ba-replication: audit, mechanism,
replication training, evaluation) and wkf1 (w8-kernel-feasibility: real-kernel timing) across both worktrees' results/gpu_hours.csv.
Hard total cap 4.0 A30-h (inclusive of failures). Planning split: audit/mechanism 0.4, training 1.6, evaluation 1.3, hardware 0.3, reserve 0.4."""
import csv
import os

LEDGERS = ["/home/boa-dit-bar/results/gpu_hours.csv", "/home/boa-dit-wkf/results/gpu_hours.csv"]
TAGS = ("bar1", "wkf1")
TOTAL_CAP = 4.0
PLAN = {"bar1": 3.3, "wkf1": 0.3, "reserve": 0.4}     # bar1 = audit/mechanism 0.4 + training 1.6 + evaluation 1.3


def spent(tag=None):
    s = 0.0
    for f in LEDGERS:
        if os.path.exists(f):
            for r in csv.reader(open(f)):
                if r and r[0] in TAGS and (tag is None or r[0] == tag):
                    s += float(r[-1])
    return s


def ok(tag, est_h, reserve_h=0.0):
    """True iff the tag stays within its planning cap (+ reserve) and the handoff total stays within 4.0."""
    t, a = spent(tag), spent()
    return a + est_h + reserve_h <= TOTAL_CAP and t + est_h <= PLAN[tag] + PLAN["reserve"], {"tag_spent": t, "total_spent": a, "est": est_h}


if __name__ == "__main__":
    print({k: round(spent(k), 4) for k in TAGS}, "total", round(spent(), 4), "of", TOTAL_CAP)
