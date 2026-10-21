#!/usr/bin/env python3
"""Append the PR#21 metric errata to the frozen pre-registration WITHOUT editing any existing field."""
import hashlib, json, os, shutil, tempfile

PREREG = "/home/T7/ojh/robot_actual/r11/mission/R15_GATE_PREREGISTRATION.json"
ERRATA = "/home/dgut/authoritative/prereg_errata.json"
BACKUP = "/home/dgut/authoritative/R15_GATE_PREREGISTRATION.orig.json"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

pre = json.load(open(PREREG))
pre_sha = sha(PREREG)
if not os.path.exists(BACKUP):
    shutil.copy2(PREREG, BACKUP)
errata = json.load(open(ERRATA))
errata["pre_sha256"] = pre_sha
if "errata" in pre:
    pre["errata"] = [e for e in pre["errata"] if e.get("date_utc") != errata["date_utc"]]
    pre["errata"].append(errata)
else:
    pre["errata"] = [errata]
d = os.path.dirname(PREREG)
fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-prereg-", suffix=".json")
with os.fdopen(fd, "w") as fh:
    json.dump(pre, fh, indent=1)
    fh.flush(); os.fsync(fh.fileno())
os.replace(tmp, PREREG)
post = json.load(open(PREREG))
# prove that nothing above the errata block changed
strip = lambda x: {k: v for k, v in x.items() if k != "errata"}
assert strip(post) == strip(json.load(open(BACKUP))), "existing fields were modified - aborting"
print("pre_sha256 ", pre_sha)
print("post_sha256", sha(PREREG))
print("errata items", len(post["errata"]), "fields_intact", True)
print("gates untouched:", sorted(post["gates"].keys()))
print("cv_macro_l1 unchanged:", post["cv_macro_l1"])
