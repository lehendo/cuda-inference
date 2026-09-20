#!/usr/bin/env python3
"""Print a compact table of gie_bench results.jsonl rows (for quick inspection on the cluster)."""
import json
import sys

for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        print(f"{r.get('workload','')[:28]:28s} {r.get('precision',''):5s} {r.get('fusion',''):8s} {r.get('backend',''):14s} {r.get('mode',''):8s} "
              f"{r.get('status',''):18s} {r.get('median_us',0):10.1f} us {r.get('tflops',0):7.2f} TF  {str(r.get('kernel',''))[:42]:42s} {str(r.get('reason',''))[:90]}")
