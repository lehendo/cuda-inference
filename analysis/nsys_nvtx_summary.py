#!/usr/bin/env python3
"""Attribute GPU kernels in an Nsight Systems sqlite export to NVTX ranges (stdlib only).

usage: nsys_nvtx_summary.py <report.sqlite> list                      # NVTX range names with counts and total time
       nsys_nvtx_summary.py <report.sqlite> kernels <name-substring>  # kernels launched inside matching ranges
       nsys_nvtx_summary.py <report.sqlite> outside <name-substring>  # kernels NOT inside matching ranges (setup/tuning)
       nsys_nvtx_summary.py <report.sqlite> after <end-of-range> <name-substring>
                                                                      # like 'kernels', but only launches after the END of the
                                                                      # first range matching <end-of-range> (e.g. the autotuner:
                                                                      # steady-state launches only)
Export with:  nsys export --type sqlite -o report.sqlite report.nsys-rep
A kernel belongs to an NVTX range when its CUDA API launch call (matched by correlationId) started inside the range on the
same thread; the kernel's own GPU time is what is summed. Nothing here uses hardware counters.
"""
import sqlite3
import sys
from collections import defaultdict


def connect(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    db = connect(sys.argv[1])
    mode = sys.argv[2]
    after = None
    if mode == "after":
        end_pat = sys.argv[3]
        after = min((e for s_, e, t, _ in db.cursor().execute(
            "select start, end, coalesce(text, (select value from StringIds where id=textId)), globalTid from NVTX_EVENTS where end is not null").fetchall()
            if end_pat in (t or "")), default=None)
        if after is None:
            print(f"no NVTX range matching '{end_pat}'")
            return 1
        mode = "kernels"
        pat = sys.argv[4]
    else:
        pat = sys.argv[3] if len(sys.argv) > 3 else ""
    cur = db.cursor()
    tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
    nv = cur.execute("select start, end, coalesce(text, (select value from StringIds where id=textId)), globalTid from NVTX_EVENTS where end is not null").fetchall()
    if mode == "list":
        agg = defaultdict(lambda: [0, 0])
        for s, e, t, _ in nv:
            agg[t][0] += 1
            agg[t][1] += e - s
        for t, (n, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:60]:
            print(f"{n:6d}  {d / 1e3:12.1f} us total  {t}")
        return 0
    runtime = cur.execute("select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    corr_launch = {c: (s, g) for c, s, g in runtime}
    kern = cur.execute("select correlationId, start, end, (select value from StringIds where id=demangledName) from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    ranges = [(s, e, t, g) for s, e, t, g in nv if pat in (t or "")]
    by_thread = defaultdict(list)
    for s, e, t, g in ranges:
        by_thread[g].append((s, e))
    agg = defaultdict(lambda: [0, 0])
    total_in = total_out = 0
    for c, ks, ke, name in kern:
        ls, g = corr_launch.get(c, (None, None))
        inside = ls is not None and any(s <= ls <= e for s, e in by_thread.get(g, [])) and (after is None or ls > after)
        if (mode == "kernels") == inside:
            agg[name][0] += 1
            agg[name][1] += ke - ks
        if inside:
            total_in += ke - ks
        else:
            total_out += ke - ks
    print(f"kernel time inside ranges matching '{pat}': {total_in / 1e3:.1f} us; outside: {total_out / 1e3:.1f} us")
    for name, (n, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:25]:
        print(f"{n:6d}  {d / 1e3:12.1f} us total  {d / n / 1e3:10.1f} us avg  {name[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
