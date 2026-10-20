"""R8 normalization contract: positional bounds (legacy) vs signed motion scaling.

Why this module exists
----------------------
`r8_contract.build_proprio` normalizes the THREE slots (state, velocity, acceleration)
with the SAME positional bounds pmin/pmax. Measured on the real Push-T HDF5 this pins
~31% of all V/Acc entries to -1, maps ZERO motion to a non-zero value, and in 5 of the
14 dimensions collapses every negative velocity into a single value (see
report/R8_PROPRIO_NORMALIZATION_AUDIT.md).

Modes
-----
  LEGACY_POSITION   state -> (2(x-pmin)/(pmax-pmin)-1); V, Acc -> the SAME positional map
                    (bit-compatible with r8_contract.build_proprio, kept as legacy)
  SIGNED_MOTION     state -> positional map (unchanged); V -> v/s_v, Acc -> a/s_a
                    with TRAIN-only scales s_d = max(q(|x_d|, q), eps)
                    => zero motion maps to exactly 0.0 and the sign is preserved

All scales are fitted on TRAIN episodes only and serialized into checkpoint metadata so
that training and inference cannot diverge.
"""
import numpy as np

BASE_DIM = 14
N_DIM = 42
LEGACY_POSITION = "legacy_position"
SIGNED_MOTION = "signed_motion"
MODES = (LEGACY_POSITION, SIGNED_MOTION)


def _a14(x, name):
    a = np.asarray(x, dtype=np.float32)
    if a.shape[-1] != BASE_DIM:
        raise ValueError("%s must have a last dimension of 14" % name)
    if not np.isfinite(a).all():
        raise ValueError("%s contains non-finite values" % name)
    return a


def positional_normalize(x, pmin, pmax):
    """Bit-compatible with r8_contract.normalize_block."""
    v = _a14(x, "x")
    lo, hi = np.asarray(pmin, np.float32), np.asarray(pmax, np.float32)
    if lo.shape != (BASE_DIM,) or hi.shape != (BASE_DIM,):
        raise ValueError("pmin/pmax must be 14D")
    if (hi <= lo).any():
        raise ValueError("pmax must exceed pmin")
    return np.clip(2.0 * (v - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0).astype(np.float32)


def fit_scales(velocities, accelerations, quantile=0.99):
    """TRAIN-only per-dimension scales for the two motion slots."""
    v = _a14(np.asarray(velocities), "velocities")
    a = _a14(np.asarray(accelerations), "accelerations")
    if len(v) < 100 or len(a) < 100:
        raise ValueError("need at least 100 TRAIN rows per slot")
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0,1)")
    sv = np.maximum(np.quantile(np.abs(v), quantile, axis=0).astype(np.float32), 1e-6)
    sa = np.maximum(np.quantile(np.abs(a), quantile, axis=0).astype(np.float32), 1e-6)
    return sv, sa


class NormalizationContract:
    def __init__(self, mode, pmin, pmax, v_scale=None, a_scale=None, quantile=None):
        if mode not in MODES:
            raise ValueError("unknown normalization mode %r (expected %s)" % (mode, MODES))
        self.mode = mode
        self.pmin = _a14(np.asarray(pmin), "pmin").reshape(BASE_DIM)
        self.pmax = _a14(np.asarray(pmax), "pmax").reshape(BASE_DIM)
        if (self.pmax <= self.pmin).any():
            raise ValueError("pmax must exceed pmin")
        self.quantile = quantile
        self.v_scale = None if v_scale is None else _a14(np.asarray(v_scale), "v_scale").reshape(BASE_DIM)
        self.a_scale = None if a_scale is None else _a14(np.asarray(a_scale), "a_scale").reshape(BASE_DIM)
        if mode == SIGNED_MOTION:
            if self.v_scale is None or self.a_scale is None:
                raise ValueError("signed_motion requires fitted v_scale and a_scale")
            if (self.v_scale <= 0).any() or (self.a_scale <= 0).any():
                raise ValueError("motion scales must be strictly positive")

    @classmethod
    def fit(cls, train_states, train_velocities, train_accelerations, pmin, pmax,
            mode=SIGNED_MOTION, quantile=0.99):
        sv, sa = fit_scales(train_velocities, train_accelerations, quantile)
        return cls(mode, pmin, pmax, sv, sa, quantile)

    # ------------------------------------------------------------------ transform
    def transform(self, state, velocity, acceleration):
        s = positional_normalize(state, self.pmin, self.pmax)
        if self.mode == LEGACY_POSITION:
            v = positional_normalize(velocity, self.pmin, self.pmax)
            a = positional_normalize(acceleration, self.pmin, self.pmax)
        else:
            v = np.clip(_a14(velocity, "velocity") / self.v_scale, -1.0, 1.0).astype(np.float32)
            a = np.clip(_a14(acceleration, "acceleration") / self.a_scale, -1.0, 1.0).astype(np.float32)
        out = np.concatenate([s, v, a], axis=-1).astype(np.float32)
        if out.shape[-1] != N_DIM:
            raise ValueError("expected %d features, got %s" % (N_DIM, out.shape))
        if not np.isfinite(out).all():
            raise ValueError("non-finite normalized features")
        return out

    def zero_motion_offsets(self):
        """Normalized value of ZERO velocity / acceleration (per dimension)."""
        z = np.zeros(BASE_DIM, np.float32)
        if self.mode == LEGACY_POSITION:
            off = positional_normalize(z, self.pmin, self.pmax)
            return {"velocity": off.tolist(), "acceleration": off.tolist(),
                    "is_zero": bool(np.all(np.abs(off) <= 1e-6))}
        return {"velocity": [0.0] * BASE_DIM, "acceleration": [0.0] * BASE_DIM,
                "is_zero": True}

    def sign_preserved(self, velocity, acceleration):
        """True if a negative motion never normalizes to the zero-motion value."""
        v = _a14(velocity, "velocity")
        a = _a14(acceleration, "acceleration")
        if self.mode == LEGACY_POSITION:
            zv = positional_normalize(np.zeros(BASE_DIM, np.float32), self.pmin, self.pmax)
            za = zv
            vn = positional_normalize(v, self.pmin, self.pmax)
            an = positional_normalize(a, self.pmin, self.pmax)
        else:
            zv = za = np.zeros(BASE_DIM, np.float32)
            vn = np.clip(v / self.v_scale, -1.0, 1.0)
            an = np.clip(a / self.a_scale, -1.0, 1.0)
        neg_v = v < -1e-8
        neg_a = a < -1e-8
        zv_full = np.broadcast_to(np.asarray(zv, np.float64), vn.shape)
        za_full = np.broadcast_to(np.asarray(za, np.float64), an.shape)
        v_ok = bool((not neg_v.any()) or np.all(np.asarray(vn, np.float64)[neg_v]
                                               < zv_full[neg_v]))
        a_ok = bool((not neg_a.any()) or np.all(np.asarray(an, np.float64)[neg_a]
                                               < za_full[neg_a]))
        return {"velocity": v_ok, "acceleration": a_ok}

    # ------------------------------------------------------------------- metadata
    def to_metadata(self):
        out = {"proprio_mode": "svacc", "proprio_dim": N_DIM,
               "normalization_mode": self.mode,
               "pmin": self.pmin.tolist(), "pmax": self.pmax.tolist(),
               "motion_scale_fit_source": "TRAIN episodes only"}
        if self.mode == SIGNED_MOTION:
            out["v_scale"] = self.v_scale.tolist()
            out["a_scale"] = self.a_scale.tolist()
            out["motion_quantile"] = float(self.quantile)
        return out

    @classmethod
    def from_metadata(cls, meta):
        mode = meta.get("normalization_mode")
        if mode not in MODES:
            raise ValueError("checkpoint lacks a known normalization_mode: %r" % (mode,))
        return cls(mode, meta["pmin"], meta["pmax"], meta.get("v_scale"), meta.get("a_scale"),
                   meta.get("motion_quantile"))

    def __repr__(self):
        return "NormalizationContract(mode=%s, quantile=%s)" % (self.mode, self.quantile)
