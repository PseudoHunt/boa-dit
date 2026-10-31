"""Common budget for the 2026-10-31 handoff (sr1 on sana-repeatability, dar1 / w8s1 on dit-anchor-refit): sums the three tags across both worktrees'
results/gpu_hours.csv. Total cap 4.0 A30-h; planning caps sr1 0.6, dar1 2.5, w8s1 0.4, reserve 0.5."""
import csv
import os

LEDGERS = ["/home/boa-dit-sr/results/gpu_hours.csv", "/home/boa-dit-dar/results/gpu_hours.csv"]
TAGS = ("sr1", "dar1", "w8s1")
TOTAL_CAP = 4.0
PLAN = {"sr1": 0.6, "dar1": 2.5, "w8s1": 0.4, "reserve": 0.5}


def spent(tag=None):
    s = 0.0
    for f in LEDGERS:
        if os.path.exists(f):
            for r in csv.reader(open(f)):
                if r and r[0] in TAGS and (tag is None or r[0] == tag):
                    s += float(r[-1])
    return s


def ok(tag, est_h, reserve_h=0.0):
    """True iff the tag stays within its planning cap and the handoff total stays within 4.0 (reserve usable only through the total)."""
    t, a = spent(tag), spent()
    return a + est_h + reserve_h <= TOTAL_CAP and t + est_h <= PLAN[tag] + PLAN["reserve"], {"tag_spent": t, "total_spent": a, "est": est_h}
