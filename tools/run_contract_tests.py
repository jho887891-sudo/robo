#!/usr/bin/env python3
"""Run a pytest-style test module without pytest installed.

Executes every callable named test_*, honours the __skip__ attribute set by the
shim's mark.skipif, prints one line per test and returns non-zero if any failed.
"""

import importlib.util
import os
import sys
import traceback


def load(path):
    spec = importlib.util.spec_from_file_location("contract_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["contract_tests"] = module
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) < 2:
        print("usage: run_contract_tests.py TEST_FILE")
        return 2
    path = os.path.abspath(sys.argv[1])
    module = load(path)
    tests = [(n, getattr(module, n)) for n in dir(module) if n.startswith("test_")]
    passed = failed = skipped = 0
    for name, fn in tests:
        if not callable(fn):
            continue
        if getattr(fn, "__skip__", False):
            print("SKIP %-62s %s" % (name, getattr(fn, "__skip_reason__", "")))
            skipped += 1
            continue
        try:
            fn()
        except Exception:
            print("FAIL %-62s" % name)
            traceback.print_exc()
            failed += 1
        else:
            print("PASS %-62s" % name)
            passed += 1
    print("\n%d passed, %d failed, %d skipped (%d total)" % (passed, failed, skipped, len(tests)))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
