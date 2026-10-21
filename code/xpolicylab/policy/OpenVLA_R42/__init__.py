"""R16 OpenVLA_R42: 42D [S,V,Acc] Push-T policy for the XPolicyLab RoboDojo serving tree.

The package is deliberately import-light: importing it must not need torch, CUDA or the 7B
weights, because the CPU audit lanes import it on a box whose GPU is busy.  The model class is
resolved lazily on first attribute access.

    from OpenVLA_R42 import Model            # only touches numpy
    from OpenVLA_R42.model import Model     # same object
"""
import os
import sys

__all__ = ["Model", "TEMPLATE_SOURCE", "INSTRUCTION", "contract"]
__version__ = "r16.1"

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, "..", "OpenVLA_Pure"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def __getattr__(name):
    if name in ("Model", "TEMPLATE_SOURCE", "INSTRUCTION", "LATENCY_PHASES", "CODEC_MODE"):
        from . import model as _m
        return getattr(_m, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def contract():
    """The 42D serving contract as observed by this package (no torch, no GPU)."""
    from . import model as _m
    import r42_adapter as _R
    import r42_env_io as _e
    return {"package": __name__,
            "version": __version__,
            "codec_mode": _m.CODEC_MODE,
            "action_token_window": list(_R.CODEC_WINDOW),
            "n_action_tokens": _R.N_ACTIONS,
            "latency_phases": list(_m.LATENCY_PHASES),
            "template_source": _m.TEMPLATE_SOURCE,
            "env_io": _e.contract()}
