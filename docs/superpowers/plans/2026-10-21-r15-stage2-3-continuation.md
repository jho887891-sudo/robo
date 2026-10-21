# R15 Stage 2/3 True Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the one already-started R15 run from global_step 300 to 1000 (+700 steps) and then to 3000 (+2000 steps) by resuming the full-state checkpoint, and decide each gate against thresholds that were frozen before any VLA DEV prediction existed.

**Architecture:** `training/r11_train_pushT_r42.py --r15` is the single entry. It enforces the strict step plan (0→300 fresh; 300→1000 only +700; 1000→3000 only +2000), seeds before model construction, books every optimizer step into the append-only WAL ledger (`/home/T7/ojh/robot_actual/r15_ledger`), and writes full-state checkpoints via `training/r15_resume.py`. `diagnostics/r15_dev_eval.py` scores the frozen 16-DEV / 32-frame gate set against identity, constant velocity and TRAIN-only affine/MLP. A new `tools/r15_gate_check.py` turns the pre-registered thresholds into a machine verdict so no gate is decided by eyeballing.

**Tech Stack:** Python 3.10, torch 2.4.1+cu121, transformers 4.40.2, peft 0.11.1, h5py, numpy; single RTX A6000 48 GB; pytest.

**Spec:** `missions/R15_TRUE_CONTINUATION_EXECUTION.md@988b6bbe`, `report/R11_R15_INDEPENDENT_RESUME_LEDGER_AUDIT.md@988b6bbe`, `missions/R11_R16_DELEGATED_EXECUTION_SPEC.md@83b5794`

## Global Constraints

- Never re-run steps already booked. 0→300 is done; stage 2 books exactly 700, stage 3 exactly 2000; total formal steps cap 3000, aggregate GPU cap 115200 s, R15 bucket cap 50400 s.
- Never resume from the R14 overfit weights, never train on DEV, never read the old 20 holdout or any new FINAL for selection.
- Every GPU job: `time_bucket` guard before start, `job_begin`/`step_commit`/`job_end` events, real `nvidia-smi` snapshots, raw stderr kept, and one GPU job at a time.
- `--save_every` must be > 0 and <= 100 (crash-recovery checkpoints).
- G5 threshold, frozen in `mission/R15_GATE_PREREGISTRATION.json`: DEV macro physical L1 <= 0.002005 (5 percent better than constant velocity 0.002110443303536158).
- Weights are never committed (`runs/`, `*.safetensors`, `*.pt` are gitignored); only JSON/log/NPZ evidence + SHA256SUMS.
- No GitHub API credential exists on the box: the Draft PR must be opened by the research lead from the pushed branch; do not advertise a `/pull/new/` URL.

---

### Task 1: Machine gate checker (thresholds as data, verdict as output)

**Files:**
- Create: `tools/r15_gate_check.py`
- Test: `tests/test_r15_gate_check.py`

**Interfaces:**
- Consumes: `mission/R15_GATE_PREREGISTRATION.json` (keys `cv_macro_l1`, `gates`), a `dev_eval.json` produced by `diagnostics/r15_dev_eval.py` (keys `models[name].metrics.physical_l1_macro`, `.still_action_rate`, `.magnitude_ratio`, `.motion_strata`, `.joint_clip_rate`), and a `stage_evidence.json` (keys `verdict`, `global_step_reached`, `target`).
- Produces: `check(gate, dev_eval, stage_evidence, prereg) -> {"gate": str, "verdict": "PASS"|"FAIL", "conditions": [{"id","ok","detail"}]}`.

- [ ] **Step 1: Write the failing test**

```python
import json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import r15_gate_check as G

PREREG = {"cv_macro_l1": 0.002110443303536158,
          "gates": {"G5_1000": {"improvement_ratio": 0.05}}}

def _dev(vla_l1, still=0.0, mag=1.0, strata=None):
    return {"models": {"constant_velocity_A=S+V": {"status": "OK",
                        "metrics": {"physical_l1_macro": 0.002110443303536158}},
                       "vla_r42": {"status": "OK", "metrics": {
                           "physical_l1_macro": vla_l1, "still_action_rate": still,
                           "magnitude_ratio": mag,
                           "motion_strata": strata or {"q1": {"l1": 0.002}, "q2": {"l1": 0.002},
                                                       "q3": {"l1": 0.002}, "q4": {"l1": 0.002}}}}}}

def test_g5_passes_at_or_below_threshold():
    r = G.check("G5_1000", _dev(0.002005), {"verdict": "REACHED_TARGET"}, PREREG)
    assert r["verdict"] == "PASS"

def test_g5_fails_above_threshold():
    r = G.check("G5_1000", _dev(0.002134), {"verdict": "REACHED_TARGET"}, PREREG)
    assert r["verdict"] == "FAIL"

def test_g5_fails_on_still_action_collapse():
    r = G.check("G5_1000", _dev(0.001, still=1.0), {"verdict": "REACHED_TARGET"}, PREREG)
    assert r["verdict"] == "FAIL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/dgut/pusht_runtime/venv_oft/bin/python -m pytest tests/test_r15_gate_check.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'r15_gate_check'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Machine verdict for the pre-registered R15 gates (no eyeballing)."""
import argparse, json, os, sys

def _m(dev, name):
    return (dev.get("models", {}).get(name) or {}).get("metrics")

def check(gate, dev, stage, prereg):
    cv = float(prereg["cv_macro_l1"])
    conds = []
    if gate == "G4_300":
        vla = _m(dev, "vla_r42")
        idt = _m(dev, "identity_A=S")
        conds = [
            {"id": "stage_reached", "ok": stage.get("verdict") == "REACHED_TARGET"},
            {"id": "beats_identity",
             "ok": float(vla["physical_l1_macro"]) < float(idt["physical_l1_macro"]),
             "detail": "%r < %r" % (vla["physical_l1_macro"], idt["physical_l1_macro"])},
            {"id": "no_still_collapse", "ok": float(vla["still_action_rate"]) < 1.0},
            {"id": "magnitude_sane", "ok": 0.2 <= float(vla["magnitude_ratio"]) <= 5.0},
        ]
    elif gate == "G5_1000":
        vla = _m(dev, "vla_r42")
        thr = cv * (1.0 - float(prereg["gates"]["G5_1000"]["improvement_ratio"]))
        strata = vla.get("motion_strata") or {}
        beats = [k for k, v in strata.items()
                 if v.get("l1") is not None and float(v["l1"]) < thr]
        conds = [
            {"id": "stage_reached", "ok": stage.get("verdict") == "REACHED_TARGET"},
            {"id": "macro_l1_vs_cv", "ok": float(vla["physical_l1_macro"]) <= thr,
             "detail": "%r <= %r" % (vla["physical_l1_macro"], thr)},
            {"id": "three_strata_or_macro",
             "ok": (float(vla["physical_l1_macro"]) <= thr) or len(beats) >= 3,
             "detail": "strata beating threshold: %s" % sorted(beats)},
            {"id": "no_still_collapse", "ok": float(vla["still_action_rate"]) < 1.0},
        ]
    elif gate == "G6_3000":
        vla = _m(dev, "vla_r42")
        conds = [
            {"id": "stage_reached", "ok": stage.get("verdict") == "REACHED_TARGET"},
            {"id": "cumulative_steps",
             "ok": int(stage.get("global_step_reached", -1)) == 3000},
            {"id": "no_still_collapse", "ok": float(vla["still_action_rate"]) < 1.0},
        ]
    else:
        raise SystemExit("unknown gate %r" % gate)
    return {"gate": gate, "verdict": "PASS" if all(c["ok"] for c in conds) else "FAIL",
            "conditions": conds,
            "thresholds": {"cv_macro_l1": cv}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["G4_300", "G5_1000", "G6_3000"])
    ap.add_argument("--dev-eval", required=True)
    ap.add_argument("--stage-evidence", default="")
    ap.add_argument("--prereg", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mission", "R15_GATE_PREREGISTRATION.json"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = json.load(open(a.dev_eval))
    stage = json.load(open(a.stage_evidence)) if a.stage_evidence else {"verdict": "REACHED_TARGET"}
    r = check(a.gate, dev, stage, json.load(open(a.prereg)))
    print(json.dumps(r, indent=1))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    return 0 if r["verdict"] == "PASS" else 3

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/dgut/pusht_runtime/venv_oft/bin/python -m pytest tests/test_r15_gate_check.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tools/r15_gate_check.py tests/test_r15_gate_check.py
git commit -m "feat(r15): machine verdict for the pre-registered G4/G5/G6 gates"
```

---

### Task 2: Stage 2 GPU run (300 -> 1000, exactly +700)

**Files:**
- Modify: nothing (entry already supports it)
- Evidence: `evidence/r15_stage1000_<utc>/`

**Interfaces:**
- Consumes: `runs/r15_stage_300_20261021T030423Z/checkpoints/step300` (manifest-verified full state), `--save_every 50`.
- Produces: `runs/r15_stage_1000_<utc>/checkpoints/step{350..1000}`, `stage_evidence.json` with `verdict == "REACHED_TARGET"`, ledger projection with `formal_steps_used == 1000`.

- [ ] **Step 1: Pre-flight the guard**
Run: `python -c "import sys;sys.path.insert(0,'training');import r15_ledger as L;led=L.Ledger();print(led.totals()['formal_steps_used'])"`
Expected: `300`

- [ ] **Step 2: Launch (one GPU job, bounded)**
```bash
nohup env PYTHONPATH=<oft>:<oft>/vla-scripts <py> training/r11_train_pushT_r42.py \
  --r15 --target-global-step 1000 --resume-from runs/r15_stage_300_<utc>/checkpoints/step300 \
  --out runs/r15_stage_1000_<utc> --bucket R15_TRAIN_FORMAL --run-id r15-s1000-<utc> \
  --est-seconds-per-step 3.0 --save_every 50 --time_margin 120 --lr 5e-4 --seed 0 > logs/... 2>&1 &
```
Expected log: `resumed from ... at global_step=300 (next frame ...)` then `step=310 ...`

- [ ] **Step 3: Verify the resume really continued**
Check `stage_evidence.json`: `start_global_step == 300`, `steps == 700`, `sampler_order_hash` identical to the stage-1 hash `f1024210752cb42e...`.
Expected: all three hold (a fresh restart would show `start_global_step == 0`).

- [ ] **Step 4: Commit evidence**

---

### Task 3: Stage-2 DEV evaluation and the G5 verdict

**Files:**
- Evidence: `evidence/r15_stage1000_<utc>/preds_dev1000.npz`, `dev_eval.json`, `gate_G5.json`

- [ ] **Step 1: Emit DEV predictions from the step1000 checkpoint**
```bash
<py> training/r11_train_pushT_r42.py --dev-preds runs/r15_stage_1000_<utc>/preds_dev1000.npz \
  --resume-from runs/r15_stage_1000_<utc>/checkpoints/step1000 --dev-per-episode 2 --dev-seed 0 \
  --out runs/r15_stage_1000_<utc>
```
Expected: `DEV predictions (32 frames, global_step=1000) -> ...`

- [ ] **Step 2: Score against the frozen baselines**
```bash
<py> diagnostics/r15_dev_eval.py --pred-npz .../preds_dev1000.npz --per-episode 2 --seed 0 \
  --out runs/r15_dev_step1000
```
Expected: `dev_frames_sha256 == 67b4320420249aef...` (a mismatch would abort)

- [ ] **Step 3: Machine verdict**
```bash
<py> tools/r15_gate_check.py --gate G5_1000 --dev-eval runs/r15_dev_step1000/dev_eval.json \
  --stage-evidence runs/r15_stage_1000_<utc>/stage_evidence.json --out evidence/.../gate_G5.json
```
Expected: `verdict` printed; exit 0 = PASS, exit 3 = FAIL.

- [ ] **Step 4: Branch on the verdict**
PASS -> Task 4. FAIL -> stop formal training, keep the checkpoint, write the diagnosis into `report/R15_STAGE_READINESS.md` (do not spend the remaining 2000 steps).

- [ ] **Step 5: Commit**

---

### Task 4: Stage 3 GPU run (1000 -> 3000, exactly +2000) — only if G5 PASS

**Files:**
- Evidence: `evidence/r15_stage3000_<utc>/`

- [ ] **Step 1: Resume from step1000 only**
Same command as Task 2 with `--target-global-step 3000 --resume-from .../checkpoints/step1000`; the entry refuses unless `start_global_step == 1000`.
Expected: `steps == 2000` in `stage_evidence.json`.

- [ ] **Step 2: DEV predictions + eval + `G6_3000` verdict** (same three commands as Task 3 with step3000).

- [ ] **Step 3: Select the DEV-best checkpoint among 300/1000/3000** and record the reason; the new FINAL data stays untouched.

- [ ] **Step 4: Commit evidence**

---

### Task 5: Report, manifest and ROUTE append

**Files:**
- Modify: `report/R15_STAGE_READINESS.md`, `mission/R11_R16_MISSION_MANIFEST.json`, `ROUTE.md`

- [ ] **Step 1: Update the manifest with real numbers** (GPU seconds per bucket from the WAL projection, formal steps, per-stage DEV macro L1, gate verdicts, checkpoint SHAs).
- [ ] **Step 2: Append to `ROUTE.md`** only what actually happened, including any FAIL and the diagnosis.
- [ ] **Step 3: Push the branch** and report the stop point (Draft PR still needs a credential).

## Self-Review

- **Spec coverage:** cumulative resume (§Lane T) = Tasks 2 and 4; DEV evaluator and baselines (§Lane E) already implemented and reused in Task 3; gate thresholds pre-registered and machine-checked (Task 1); budget guard and step cap (§Lane T4) already implemented in `r15_ledger.py`; Lane D and Lane R16 are out of scope here by the scope check and are dispatched as separate plans.
- **Placeholder scan:** no TBD/TODO; every code step contains runnable code; commands use explicit paths with `<utc>` placeholders only where the timestamp is unknowable in advance, and each is paired with the expected output that proves success.
- **Type consistency:** `check(gate, dev, stage, prereg)` is used identically in the tests, the CLI and Task 3; `dev_eval.json`/`stage_evidence.json` key paths match what `diagnostics/r15_dev_eval.py` and the `--r15` entry actually write.
