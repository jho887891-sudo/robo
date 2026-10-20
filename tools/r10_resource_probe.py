#!/usr/bin/env python3
"""R10 resource probe: record free VRAM (GiB, 2**30) and foreign GPU processes. Read-only."""
import json
import subprocess
import sys
import time


def sh(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {"cmd": cmd, "rc": p.returncode, "out": p.stdout.strip(), "err": p.stderr.strip()[:500]}


def main():
    import torch
    free, total = torch.cuda.mem_get_info()
    info = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "free_gib": free / 2 ** 30, "total_gib": total / 2 ** 30,
            "used_gib": (total - free) / 2 ** 30,
            "min_free_gib_required": 24.0,
            "gate": "PASS" if free / 2 ** 30 >= 24.0 else "BLOCKED_RESOURCE",
            "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader"]),
            "compute_apps": sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                "--format=csv,noheader"]),
            "docker_ps": sh(["docker", "ps", "--format", "{{.Names}} {{.Image}} {{.Status}}"]),
            "note": "never stop or signal a foreign process; free VRAM is whatever is left"}
    json.dump(info, open(sys.argv[1], "w"), indent=1)
    print("free_gib=%.3f total_gib=%.3f required=%.1f -> %s"
          % (info["free_gib"], info["total_gib"], info["min_free_gib_required"], info["gate"]))
    print(info["compute_apps"]["out"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
