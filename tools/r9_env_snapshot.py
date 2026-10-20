#!/usr/bin/env python
"""R9 environment + data identity snapshot (Gate A/B/C/D preflight).

Writes one JSON with code SHA, package versions, GPU occupancy at start and content
hashes of every dataset artefact the gates depend on. Read-only, no GPU work.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
HDF5 = os.environ.get("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
BOUNDS = os.environ.get("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return {"cmd": cmd, "rc": out.returncode, "stdout": out.stdout.strip(),
                "stderr": out.stderr.strip()[:2000]}
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc)}


def main():
    out_path = sys.argv[1]
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkgs = {}
    for mod in ("torch", "transformers", "peft", "h5py", "numpy", "PIL", "timm"):
        try:
            pkgs[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:  # noqa: BLE001
            pkgs[mod] = None
    episodes = (sorted(f for f in os.listdir(HDF5) if f.endswith(".hdf5"))
                if os.path.isdir(HDF5) else [])
    split_path = os.path.join(repo, "data", "split80.json")
    if not os.path.exists(split_path):
        split_path = os.path.join(os.path.dirname(repo), "data", "split80.json")
    info = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "repo": repo,
        "git": {
            "head": sh(["git", "-C", repo, "rev-parse", "HEAD"])["stdout"],
            "branch": sh(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"])["stdout"],
            "describe": sh(["git", "-C", repo, "describe", "--always", "--dirty"])["stdout"],
            "status_porcelain": sh(["git", "-C", repo, "status", "--porcelain"])["stdout"].splitlines(),
        },
        "packages": pkgs,
        "gpu": {
            "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,"
                                         "driver_version", "--format=csv,noheader"]),
            "compute_apps": sh(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                "--format=csv,noheader"]),
        },
        "data": {
            "hdf5_dir": HDF5,
            "n_episodes": len(episodes),
            "episodes_first": episodes[:3],
            "episodes_last": episodes[-3:],
            "episode_list_sha256": hashlib.sha256("\n".join(episodes).encode()).hexdigest(),
            "bounds_path": BOUNDS,
            "bounds_sha256": sha256(BOUNDS) if os.path.exists(BOUNDS) else None,
            "split_path": split_path,
            "split_sha256": sha256(split_path) if os.path.exists(split_path) else None,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps({"git": info["git"], "packages": info["packages"]}, indent=1))
    print("n_episodes", info["data"]["n_episodes"], "| episode_list_sha256",
          info["data"]["episode_list_sha256"][:16], "| bounds",
          (info["data"]["bounds_sha256"] or "?")[:16], "| split",
          (info["data"]["split_sha256"] or "?")[:16])
    print("->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
