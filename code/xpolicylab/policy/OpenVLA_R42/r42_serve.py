#!/usr/bin/env python3
"""R16: start the OpenVLA_R42 policy server (split mode) without touching the RoboDojo tree.

The RoboDojo entry point (XPolicyLab/setup_policy_server.py:31-32) resolves the model as
XPolicyLab.policy.<NAME>.model, i.e. it needs this directory to live inside the RoboDojo serving
tree.  Split mode (scripts/robodojo.sh client against an already running server) does not: only
the server needs the model.  This launcher builds the same PolicyServer straight from the r11
repo, with

    PYTHONPATH = <RoboDojo>:<RoboDojo>/XPolicyLab:<r11>/code/xpolicylab/policy

so both the client protocol package (client_server.ws) and this package (OpenVLA_R42) import.

Usage:
    python r42_serve.py --config_path deploy.yml [--host H] [--port P]
                        [--overrides key=value ...] [--print-contract]
"""
import argparse
import ast
import asyncio
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_POLICY_ROOT = os.path.dirname(_HERE)          # so "import OpenVLA_R42" resolves
for _p in (_POLICY_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _bootstrap_paths():
    """Make client_server.ws (XPolicyLab tree) importable without pip-installing it."""
    tried = []
    root = os.environ.get("ROBODOJO_ROOT", "/home/T7/ojh/robot_actual/RoboDojo")
    for cand in (root, os.path.join(root, "XPolicyLab")):
        tried.append(cand)
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.append(cand)
    return tried


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise SystemExit("config must be a YAML mapping: %s" % path)
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OpenVLA_R42 42D policy server (ws protocol)")
    p.add_argument("--config_path", "--config-path", dest="config_path", required=True)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--overrides", nargs=argparse.REMAINDER, default=None,
                   help="key=value pairs merged into the config")
    p.add_argument("--print-contract", action="store_true",
                   help="print the serving contract as json and exit (no server, no GPU)")
    return p.parse_args(argv)


def _apply_overrides(cfg, tokens):
    def parse_val(s):
        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    tokens = list(tokens or [])
    if tokens and all(("=" in t and not t.startswith("-")) for t in tokens):
        pairs = [t.split("=", 1) for t in tokens]
    else:
        if len(tokens) % 2 != 0:
            raise SystemExit("--overrides expects key=value or key value pairs, got %r" % tokens)
        pairs = list(zip(tokens[0::2], tokens[1::2]))
    for key, value in pairs:
        cfg[str(key).lstrip("-")] = parse_val(value)
    return cfg


def main(argv=None):
    args = parse_args(argv)
    tried = _bootstrap_paths()
    # argparse.REMAINDER swallows later flags, so accept --print-contract on either side
    overrides = list(args.overrides or [])
    if "--print-contract" in overrides:
        overrides.remove("--print-contract")
        args.print_contract = True
    cfg = _apply_overrides(load_config(args.config_path), overrides)
    if args.host is not None:
        cfg["host"] = args.host
    if args.port is not None:
        cfg["port"] = args.port
    for key in ("policy_name", "host", "port"):
        if not cfg.get(key):
            raise SystemExit("%s must be specified in config or overrides" % key)

    try:
        from OpenVLA_R42.model import Model
    except ImportError:                     # running with only this directory on sys.path
        from model import Model

    model = Model(cfg)
    if args.print_contract:
        print(json.dumps(model.contract(), indent=1, sort_keys=True))
        return 0
    try:
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "client_server is not importable (tried %s): set ROBODOJO_ROOT or PYTHONPATH to the "
            "RoboDojo checkout. Original error: %s" % (tried, exc))
    server = PolicyServer(model, PolicyServerConfig(
        host=cfg["host"], port=int(cfg["port"]),
        ws_ping_interval_s=cfg.get("ws_ping_interval_s", 20.0),
        ws_ping_timeout_s=cfg.get("ws_ping_timeout_s", 20.0)))
    print("[OpenVLA_R42] serving on ws://%s:%s (policy=%s)"
          % (cfg["host"], cfg["port"], cfg["policy_name"]), flush=True)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("\n[OpenVLA_R42] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
