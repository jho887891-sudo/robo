#!/usr/bin/env python3
"""R43: official GenerationMixin (+ KV cache) adapter for the R42 14-token decoder.

Why this module exists
----------------------
The R42 deployment decoder (r42_deploy_server.py:285) runs 14 *full* multimodal forwards per
control step.  R16 tried to short-circuit that by calling the language model directly
(prefill_cached/step_cached, r42_deploy_server.py:482-514), which bypasses
prepare_inputs_for_generation completely; it diverged in the free-running token sequence and was
therefore kept behind R42_KV_CACHE=1 and never used by default.

R43 asks whether the *official* Hugging Face generation stack can do the same job.  It can only
do so if three R42-specific facts travel through model.generate():

  1. proprio (42D) and proprio_projector are forward-only arguments.  The stock
     PrismaticForConditionalGeneration.prepare_inputs_for_generation
     (modeling_prismatic.py:678-713) returns only input_ids / attention_mask / pixel_values /
     past_key_values / use_cache, so proprio is silently dropped on the prefill and the model
     would run with proprio=None - a computation that differs from training.
  2. labels is REQUIRED on the multimodal prefill by the pure-autoregressive input policy
     (pure_autoreg_mode.py:22-27 raises when labels is None) and is FORBIDDEN on the incremental
     steps (modeling_prismatic.py:537 asserts labels is None).
  3. The 14 action tokens must stay inside 31744..31999 (a logits constraint) and the generation
     must not stop early.

This module is the minimal, testable layer that satisfies those three: it installs a replacement
prepare_inputs_for_generation on ONE model instance (never on the class, never in the shared
environment), installs a legal-window LogitsProcessor, and enforces the 14-token contract on the
result.  It changes no weight, no precision, no prompt and no action-decoding maths.
"""
import time

import numpy as np

TOKEN_BEGIN_IDX = 31743          # r42_adapter.TOKEN_BEGIN_IDX
LEGAL_WINDOW = (31744, 31999)    # r42_adapter.CODEC_WINDOW
N_ACTIONS = 14


class R42GenerateError(RuntimeError):
    """The official generation path violated the R42 prefill/decode contract."""


# --------------------------------------------------------------------------- logits constraint
def legal_window_logits_processor(window=LEGAL_WINDOW):
    """Mask every logit outside the 256 legal action tokens to -inf.

    A LogitsProcessor (the official extension point) is used rather than an argmax over a slice,
    so the *generation loop itself* can never emit EOS, PAD or any other vocabulary token.
    """
    from transformers import LogitsProcessor

    lo, hi = int(window[0]), int(window[1])

    class _LegalWindow(LogitsProcessor):
        def __call__(self, input_ids, scores):
            import torch
            masked = torch.full_like(scores, float("-inf"))
            masked[:, lo:hi + 1] = scores[:, lo:hi + 1]
            return masked

    return _LegalWindow()


# ------------------------------------------------------- official prepare_inputs replacement
def prepare_inputs_for_generation_impl(self, input_ids=None, past_key_values=None,
                                       inputs_embeds=None, pixel_values=None,
                                       attention_mask=None, **kwargs):
    """Two-phase contract for the R42 multimodal prefill + incremental decode.

    phase 1 (past_key_values is None): full prompt, all three cameras, 42D proprio + projector,
        and labels (the pure-autoreg input policy needs a non-None labels tensor).
    phase 2 (past_key_values is not None): ONE new token, no pixels, no proprio, no labels -
        exactly what modeling_prismatic.py:534-550 asserts.
    """
    if inputs_embeds is not None:
        raise R42GenerateError(
            "inputs_embeds prefill is not part of the R42 contract: generation must start from "
            "input_ids so that VisionBackbone + proprio_projector + label mask run inside forward")

    if past_key_values is None:
        if input_ids is None:
            raise R42GenerateError("prefill requires input_ids")
        if pixel_values is None:
            raise R42GenerateError("prefill requires pixel_values (the 3-camera observation)")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "proprio": kwargs.get("proprio"),
            "proprio_projector": kwargs.get("proprio_projector"),
            "labels": kwargs.get("labels"),
            "past_key_values": None,
            "use_cache": bool(kwargs.get("use_cache", True)),
        }

    # Incremental decode: only the newest token may be new; everything else must come from cache.
    if input_ids is None or input_ids.shape[1] < 1:
        raise R42GenerateError("incremental decode requires at least one input token")
    if pixel_values is not None:
        # generate() re-supplies every model_kwarg on every step, so the pixels MUST be dropped
        # here.  They are counted instead of ignored: adapter.generate() refuses to report success
        # unless exactly n_actions-1 steps arrived with pixels and none was forwarded.
        self._r43_pixel_drops = int(getattr(self, "_r43_pixel_drops", 0)) + 1
    return {
        "input_ids": input_ids[:, -1:],
        "attention_mask": None,
        "pixel_values": None,
        "proprio": None,
        "proprio_projector": None,
        "labels": None,
        "past_key_values": past_key_values,
        "use_cache": bool(kwargs.get("use_cache", True)),
    }


# ------------------------------------------------------------------------------- the adapter
class R42OfficialGenerateAdapter:
    """Drive the R42 14-token decode through model.generate(use_cache=True).

    The adapter installs two instance-level patches and nothing else:
      * prepare_inputs_for_generation (see prepare_inputs_for_generation_impl)
      * a counting/timing wrapper around model.forward (instrumentation only)
    uninstall() restores both, so a single process can measure the full-recompute path and the
    official generation path on the same weights.
    """

    def __init__(self, model, projector, device="cuda", token_begin_idx=TOKEN_BEGIN_IDX,
                 window=LEGAL_WINDOW, n_actions=N_ACTIONS):
        self.model = model
        self.projector = projector
        self.device = device
        self.token_begin_idx = int(token_begin_idx)
        self.window = (int(window[0]), int(window[1]))
        self.n_actions = int(n_actions)
        self._orig_prepare = None
        self._orig_forward = None
        self._installed = False
        self.calls = []
        self.vision_calls = 0

    # ------------------------------------------------------------------ install / uninstall
    def install(self):
        if self._installed:
            return self
        if not hasattr(self.model, "prepare_inputs_for_generation"):
            raise R42GenerateError("model has no prepare_inputs_for_generation")
        import types
        self._orig_prepare = self.model.prepare_inputs_for_generation
        self.model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation_impl, self.model)
        self._orig_forward = self.model.forward
        self.model.forward = self._counting_forward(self._orig_forward)
        self._installed = True
        return self

    def uninstall(self):
        if not self._installed:
            return self
        self.model.prepare_inputs_for_generation = self._orig_prepare
        self.model.forward = self._orig_forward
        self._installed = False
        return self

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False

    # ------------------------------------------------------------------------ instrumentation
    def _counting_forward(self, orig):
        import functools

        @functools.wraps(orig)          # keep the forward signature: generate() validates
        def wrapped(*args, **kwargs):   # model_kwargs against it (utils.py:1130)
            return self._timed_forward(orig, args, kwargs)

        return wrapped

    def _timed_forward(self, orig, args, kwargs):
        t0 = time.perf_counter()
        out = orig(*args, **kwargs)
        dt_ms = (time.perf_counter() - t0) * 1e3
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        self.calls.append({
            "input_len": int(ids.shape[1]) if ids is not None else None,
            "has_pixel_values": kwargs.get("pixel_values") is not None,
            "has_proprio": kwargs.get("proprio") is not None,
            "has_labels": kwargs.get("labels") is not None,
            "has_cache": kwargs.get("past_key_values") is not None,
            "ms": dt_ms,
        })
        return out

    def count_vision(self, core):
        """Wrap the vision backbone so 'one encode per request' is measured, not assumed."""
        vb = getattr(core, "vision_backbone", None)
        if vb is None:
            raise R42GenerateError("no vision_backbone to instrument")
        orig = vb.forward

        def wrapped(*a, **k):
            self.vision_calls += 1
            return orig(*a, **k)

        vb.forward = wrapped
        return orig

    # ------------------------------------------------------------------------------ generate
    def generate(self, prompt_ids, pixel_values, proprio, labels_ids=None):
        """One control step: prefill the multimodal prefix, then 14 legal tokens from the cache."""
        import torch
        from transformers import LogitsProcessorList

        prompt = [int(x) for x in prompt_ids]
        if not prompt:
            raise R42GenerateError("prompt_ids is empty")
        ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        pr = torch.as_tensor(np.asarray(proprio, np.float32).reshape(1, -1),
                             dtype=torch.float32, device=self.device)
        if labels_ids is None:
            labels_ids = ids.clone()
        elif not torch.is_tensor(labels_ids):
            labels_ids = torch.tensor([list(labels_ids)], dtype=torch.long, device=self.device)

        self.calls = []
        self.vision_calls = 0
        self.model._r43_pixel_drops = 0
        gen_kwargs = dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=pixel_values,
            proprio=pr,
            proprio_projector=self.projector,
            labels=labels_ids,
            max_new_tokens=self.n_actions,
            min_new_tokens=self.n_actions,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_logits=True,
            logits_processor=LogitsProcessorList(
                [legal_window_logits_processor(self.window)]),
            eos_token_id=-1,          # the logits processor already makes EOS unreachable
            pad_token_id=int(self.window[1]) + 1,
            synced_gpus=False,
        )
        # The deployed full forward runs inside this exact autocast context
        # (r42_deploy_server.py:522-529).  generate() must use it too, otherwise the bf16
        # weights meet float32 activations (first failure: q_proj mat1 float != mat2 bfloat16)
        # and, worse, the prefill numerics would differ from path A for a reason that has
        # nothing to do with caching.
        import contextlib
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if str(self.device).startswith("cuda") else contextlib.nullcontext())
        t0 = time.perf_counter()
        try:
            with ctx:
                out = self.model.generate(**gen_kwargs)
        except Exception as exc:                      # fail closed, never fall back silently
            raise R42GenerateError("official generate() failed: %s: %s"
                                   % (type(exc).__name__, exc)) from exc
        wall_ms = (time.perf_counter() - t0) * 1e3

        seq = out.sequences
        if seq.shape[0] != 1:
            raise R42GenerateError("generate() returned batch %d, expected 1" % seq.shape[0])
        new_tokens = [int(x) for x in seq[0, len(prompt):].tolist()]
        if len(new_tokens) != self.n_actions:
            raise R42GenerateError("generate() produced %d tokens, expected %d"
                                   % (len(new_tokens), self.n_actions))
        lo, hi = self.window
        bad = [t for t in new_tokens if not (lo <= t <= hi)]
        if bad:
            raise R42GenerateError("generate() emitted tokens outside the legal window: %r" % bad)

        logits = getattr(out, "logits", None)
        window = np.zeros((len(new_tokens), hi - lo + 1), np.float32)
        free_argmax = []
        margins = []
        if logits is not None:
            if len(logits) != len(new_tokens):
                raise R42GenerateError("got %d per-step logits for %d tokens"
                                       % (len(logits), len(new_tokens)))
            for k, step in enumerate(logits):
                row = step[0].float().cpu().numpy()
                window[k] = row[lo:hi + 1]
                free_argmax.append(int(np.argmax(row)))
                w = window[k]
                order = np.argsort(w)[-2:][::-1]
                margins.append({"argmax_bin": int(order[0]),
                                "top1": float(w[order[0]]),
                                "top2": float(w[order[1]]),
                                "margin": float(w[order[0]] - w[order[1]])})

        info = {
            "prompt_len": len(prompt),
            "n_tokens": len(new_tokens),
            "forward_calls": len(self.calls),
            "prefill_calls": sum(1 for c in self.calls if not c["has_cache"]),
            "decode_calls": sum(1 for c in self.calls if c["has_cache"]),
            "prefill_input_len": self.calls[0]["input_len"] if self.calls else None,
            "decode_input_lens": [c["input_len"] for c in self.calls[1:]],
            "prefill_ms": self.calls[0]["ms"] if self.calls else None,
            "decode_ms": [c["ms"] for c in self.calls[1:]],
            "wall_ms": wall_ms,
            "vision_calls": self.vision_calls,
            "pixel_kwargs_dropped_on_decode": int(
                getattr(self.model, "_r43_pixel_drops", 0)),
            "labels_on_prefill_only": all(
                (c["has_labels"] == (not c["has_cache"])) for c in self.calls),
            "proprio_on_prefill_only": all(
                (c["has_proprio"] == (not c["has_cache"])) for c in self.calls),
            "pixels_on_prefill_only": all(
                (c["has_pixel_values"] == (not c["has_cache"])) for c in self.calls),
        }
        # Hard contract assertions: a run that violates them is not a valid R42 decode.
        for key in ("labels_on_prefill_only", "proprio_on_prefill_only",
                    "pixels_on_prefill_only"):
            if not info[key]:
                raise R42GenerateError("generation contract violated: %s is False" % key)
        if info["pixel_kwargs_dropped_on_decode"] != self.n_actions - 1:
            raise R42GenerateError(
                "expected %d decode steps to carry (and drop) stale pixel_values, got %d"
                % (self.n_actions - 1, info["pixel_kwargs_dropped_on_decode"]))
        if info["prefill_calls"] != 1 or info["decode_calls"] != self.n_actions - 1:
            raise R42GenerateError(
                "expected exactly 1 prefill + %d incremental calls, got %d + %d"
                % (self.n_actions - 1, info["prefill_calls"], info["decode_calls"]))
        if info["vision_calls"] > 1:
            raise R42GenerateError("vision backbone ran %d times for one request (expected 1)"
                                   % info["vision_calls"])

        return {
            "ids": new_tokens,
            "bins": [t - self.token_begin_idx - 1 for t in new_tokens],
            "window_logits": window,
            "free_argmax_ids": free_argmax,
            "margins": margins,
            "info": info,
        }
