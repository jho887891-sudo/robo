#!/usr/bin/env python3
"""核对“实际部署加载的权重”与 CHECKPOINT_IDENTITY.md 记录的 SHA-256 是否一致。

对应第一轮审查问题 5。做法：
  1. 从 deploy.yml 读出线上真正加载的三个路径（base_model / adapter_dir / proprio_projector）；
  2. 从 CHECKPOINT_IDENTITY.md 的表格里解析出期望的 sha256 与字节数；
  3. 现场重算 sha256 并逐项比对，输出 MATCH / MISMATCH / UNVERIFIED；
  4. 任何一项不可读 -> 标记 UNVERIFIED（绝不写成“已验证”）。

用法：
  python diagnostics/check_checkpoint_identity.py
  python diagnostics/check_checkpoint_identity.py --identity CHECKPOINT_IDENTITY.md --deploy-yml <path>
"""
import argparse
import hashlib
import json
import os
import re
import sys

DEFAULT_IDENTITY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "CHECKPOINT_IDENTITY.md")
DEFAULT_YML = os.environ.get("PUSHT_DEPLOY_YML",
                             "/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_Pure/deploy.yml")

SHA_RE = re.compile(r"sha256\s+([0-9a-f]{64})", re.I)
SIZE_RE = re.compile(r"\*\*([0-9][0-9,]*)\s*B\*\*")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_identity(path):
    """返回 {文件名: {'sha256':..., 'size':...}}。"""
    exp = {}
    if not os.path.exists(path):
        return exp
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.startswith("|"):
            continue
        m = SHA_RE.search(line)
        if not m:
            continue
        name = None
        for cand in ("adapter_model.safetensors", "proprio_projector--pure.pt", "adapter_config.json"):
            if cand in line:
                name = cand
        if not name:
            continue
        sm = SIZE_RE.search(line)
        size = int(sm.group(1).replace(",", "")) if sm else None
        exp[name] = {"sha256": m.group(1).lower(), "size": size}
    return exp


def read_deploy(path):
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\s*([a-z_]+):\s*(\S.*?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", default=DEFAULT_IDENTITY)
    ap.add_argument("--deploy-yml", default=DEFAULT_YML)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    expected = parse_identity(args.identity)
    cfg = read_deploy(args.deploy_yml)
    print("identity file : %s" % args.identity)
    print("deploy yml    : %s" % args.deploy_yml)
    if cfg is None:
        print("DEPLOY_YML_MISSING -> 全部 UNVERIFIED")
        return 0
    targets = {
        "adapter_model.safetensors": os.path.join(cfg.get("adapter_dir", ""), "adapter_model.safetensors"),
        "adapter_config.json": os.path.join(cfg.get("adapter_dir", ""), "adapter_config.json"),
        "proprio_projector--pure.pt": cfg.get("proprio_projector_path", ""),
    }
    rows, verdict = [], "ALL_MATCH"
    for name, path in targets.items():
        exp = expected.get(name, {})
        row = {"file": name, "path": path, "expected_sha256": exp.get("sha256"),
               "expected_size": exp.get("size")}
        if not path or not os.path.exists(path):
            row.update({"status": "UNVERIFIED", "reason": "file not readable",
                        "actual_sha256": None, "actual_size": None})
            verdict = "SOME_UNVERIFIED" if verdict == "ALL_MATCH" else verdict
        else:
            actual = sha256_file(path)
            size = os.path.getsize(path)
            row.update({"actual_sha256": actual, "actual_size": size})
            if not exp:
                row.update({"status": "UNVERIFIED", "reason": "no expectation recorded"})
                verdict = "SOME_UNVERIFIED" if verdict == "ALL_MATCH" else verdict
            elif actual == exp["sha256"] and (exp["size"] in (None, size)):
                row.update({"status": "MATCH"})
            else:
                row.update({"status": "MISMATCH"})
                verdict = "MISMATCH"
        rows.append(row)

    print("\n%-30s %14s %-68s %s" % ("file", "size", "sha256(actual)", "status"))
    for r in rows:
        print("%-30s %14s %-68s %s" % (
            r["file"], r["actual_size"] if r["actual_size"] is not None else "N/A",
            r["actual_sha256"] or "N/A", r["status"]))
    print("\nVERDICT: %s" % verdict)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"deploy_yml": args.deploy_yml, "verdict": verdict, "rows": rows}, fh, indent=1)
        print("wrote " + args.out)
    print("IDENTITY_CHECK_DONE")
    return 0 if verdict in ("ALL_MATCH", "SOME_UNVERIFIED") else 1


if __name__ == "__main__":
    sys.exit(main())
