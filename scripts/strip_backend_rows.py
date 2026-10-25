#!/usr/bin/env python3
"""Remove all rows of one backend from a results directory (results.jsonl + results.csv) so the backend can be
re-measured and appended. Used only when a baseline's measurement method was corrected; the correction is noted in the
README. usage: strip_backend_rows.py <results_dir> <backend>"""
import csv
import json
import os
import sys

d, backend = sys.argv[1], sys.argv[2]
jl = os.path.join(d, "results.jsonl")
rows = [json.loads(l) for l in open(jl) if l.strip()]
keep = [r for r in rows if r.get("backend") != backend]
with open(jl, "w") as fh:
    for r in keep:
        fh.write(json.dumps(r) + "\n")
cs = os.path.join(d, "results.csv")
if os.path.exists(cs):
    with open(cs, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        bi = header.index("backend")
        body = [row for row in rd if row[bi] != backend]
    with open(cs, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(body)
print(f"{d}: removed {len(rows) - len(keep)} '{backend}' rows, kept {len(keep)}")
