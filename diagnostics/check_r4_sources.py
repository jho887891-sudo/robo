#!/usr/bin/env python3
"""Read-only R4 preflight for unifying training/deployment source files.

A successful run means the deployed policy Python files match the R4 checkout
AND the base model's source matches the pristine modeling module bundled here.
It does not validate any model weights or imply closed-loop success.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def check(repo, policy_dir, model_dir):
    repo = Path(repo)
    policy_dir = Path(policy_dir)
    model_dir = Path(model_dir)
    expected = repo / "code/xpolicylab/policy/OpenVLA_Pure"
    checks = {}
    for filename in ("model.py", "pure_autoreg_mode.py", "action_codec.py",
                     "action_target.py", "deploy_contract.py"):
        a = expected / filename
        b = policy_dir / filename
        checks["policy_" + filename] = {
            "reference": str(a),
            "deployed": str(b),
            "reference_sha256": sha(a) if a.is_file() else None,
            "deployed_sha256": sha(b) if b.is_file() else None,
        }
    reference_model = repo / "code/openvla_dynamic_modules/modeling_prismatic.py"
    active_model = model_dir / "modeling_prismatic.py"
    checks["dynamic_modeling_prismatic.py"] = {
        "reference": str(reference_model),
        "deployed": str(active_model),
        "reference_sha256": sha(reference_model) if reference_model.is_file() else None,
        "deployed_sha256": sha(active_model) if active_model.is_file() else None,
    }
    for item in checks.values():
        item["match"] = bool(item["reference_sha256"]) and (
            item["reference_sha256"] == item["deployed_sha256"])
    return {"passed": all(item["match"] for item in checks.values()), "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--policy-dir", required=True, help="RoboDojo/XPolicyLab/policy/OpenVLA_Pure")
    ap.add_argument("--base-model", required=True, help="pristine OpenVLA base model dir")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    result = check(args.repo, args.policy_dir, args.base_model)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
