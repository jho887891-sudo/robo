#!/usr/bin/env python3
"""R43 CPU regression tests for the official Generate + KV-cache adapter.

Everything here runs on CPU with a 4 MB miniature VLM that reproduces the *contract* of
PrismaticForConditionalGeneration.forward (one multimodal prefill with pixel_values + proprio +
labels, then single-token cached decode steps) and the *defect* of the stock
prepare_inputs_for_generation (it drops proprio / proprio_projector / labels).

These tests prove interface and contract logic only.  They are not evidence that the real 7B
model behaves this way - that requires the GPU run in diagnostics/r43_official_generate_kv.py.
"""
import os
import sys
import unittest

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, PretrainedConfig, PreTrainedModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POLICY = os.path.join(ROOT, "code", "xpolicylab", "policy")
for p in (os.path.join(POLICY, "OpenVLA_R42"), os.path.join(POLICY, "OpenVLA_Pure")):
    if p not in sys.path:
        sys.path.insert(0, p)

import r42_official_generate as OG   # noqa: E402

VOCAB = 32000
HIDDEN = 32
VISION_TOKENS = 4
PROMPT = [1, 5, 7, 29871]


class _Vision(nn.Module):
    def __init__(self, hidden, tokens):
        super().__init__()
        self.proj = nn.Linear(3, hidden)
        self.tokens = tokens
        self.calls = 0

    def forward(self, pixel_values):
        self.calls += 1
        b = pixel_values.shape[0]
        base = pixel_values.mean(dim=(2, 3))[:, :3]
        return self.proj(base).unsqueeze(1).expand(b, self.tokens, -1).contiguous()


class MiniConfig(PretrainedConfig):
    model_type = "mini_r42"

    def __init__(self, vocab_size=VOCAB, hidden=HIDDEN, vision_tokens=VISION_TOKENS,
                 pad_token_id=VOCAB - 1, eos_token_id=-1, **kw):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kw)
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.vision_tokens = vision_tokens


class MiniVLM(PreTrainedModel):
    """Same two-phase forward contract as modeling_prismatic.py:499-550."""

    config_class = MiniConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_backbone = _Vision(config.hidden, config.vision_tokens)
        self.language_model = LlamaForCausalLM(LlamaConfig(
            vocab_size=config.vocab_size, hidden_size=config.hidden, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, intermediate_size=config.hidden * 2,
            max_position_embeddings=512))
        self.calls = []
        self.cache_ids = []
        self.bias_token = None
        self.labels_seen = []

    # --- the stock (defective) implementation, mirrored from modeling_prismatic.py:678-713
    def prepare_inputs_for_generation(self, input_ids=None, past_key_values=None,
                                      inputs_embeds=None, pixel_values=None,
                                      attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "pixel_values": pixel_values, "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache")}

    def _process_action_masks(self, labels):
        if labels is None:
            raise ValueError("labels required for multimodal autoregressive forward")
        return torch.zeros_like(labels, dtype=torch.bool)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                labels=None, past_key_values=None, use_cache=None, proprio=None,
                proprio_projector=None, **kwargs):
        self.calls.append({"len": int(input_ids.shape[1]),
                           "pixels": pixel_values is not None,
                           "proprio": proprio is not None,
                           "labels": labels is not None,
                           "cache": past_key_values is not None})
        if past_key_values is not None:
            self.cache_ids.append(id(past_key_values))
        if input_ids.shape[1] == 1:
            assert past_key_values is not None, "cache required for a 1-token step"
            assert labels is None, "labels must be None on incremental steps"
            out = self.language_model(input_ids=input_ids, past_key_values=past_key_values,
                                      use_cache=use_cache, return_dict=True)
            return self._maybe_bias(out)
        # multimodal prefill
        self.labels_seen.append(labels is not None)
        self._process_action_masks(labels)               # raises if labels is None
        emb = self.language_model.get_input_embeddings()(input_ids)
        vision = self.vision_backbone(pixel_values)
        if proprio_projector is not None and proprio is not None:
            vision = torch.cat([vision, proprio_projector(proprio).unsqueeze(1)], dim=1)
        emb = torch.cat([emb[:, :1], vision, emb[:, 1:]], dim=1)
        mask = None
        if attention_mask is not None:
            extra = torch.ones((attention_mask.shape[0], vision.shape[1]),
                               dtype=attention_mask.dtype)
            mask = torch.cat([attention_mask[:, :1], extra, attention_mask[:, 1:]], dim=1)
        out = self.language_model(inputs_embeds=emb, attention_mask=mask,
                                  use_cache=use_cache, return_dict=True)
        self.cache_ids.append(id(out.past_key_values))
        return self._maybe_bias(out)

    def _maybe_bias(self, out):
        if self.bias_token is not None:
            out.logits[..., int(self.bias_token)] += 1e4
        return out


def make_model():
    torch.manual_seed(0)
    model = MiniVLM(MiniConfig())
    model.eval()
    return model


def make_inputs(proprio_value=0.5):
    torch.manual_seed(1)
    pixel = torch.rand(1, 3, 8, 8)
    prop = torch.full((1, 42), float(proprio_value))
    projector = nn.Linear(42, HIDDEN)
    with torch.no_grad():
        projector.weight.fill_(0.1)
        projector.bias.zero_()
    return pixel, prop, projector


class TestStockPathIsBroken(unittest.TestCase):
    def test_stock_generate_drops_proprio_and_labels(self):
        """Without the adapter the official loop silently loses the 42D state and labels."""
        model = make_model()
        pixel, prop, projector = make_inputs()
        with self.assertRaises(Exception):
            model.generate(input_ids=torch.tensor([PROMPT]),
                           attention_mask=torch.ones(1, len(PROMPT), dtype=torch.long),
                           pixel_values=pixel, proprio=prop, proprio_projector=projector,
                           labels=torch.tensor([PROMPT]), max_new_tokens=14, do_sample=False,
                           num_beams=1, use_cache=True)
        # the failure is the pure-autoreg labels contract, not a bug in this test
        self.assertEqual(model.calls[0]["labels"], False)


class TestOfficialAdapter(unittest.TestCase):
    def test_prefill_then_cached_decode(self):
        model = make_model()
        pixel, prop, projector = make_inputs()
        adapter = OG.R42OfficialGenerateAdapter(model, projector, device="cpu").install()
        res = adapter.generate(PROMPT, pixel, prop.numpy())
        self.assertEqual(len(res["ids"]), 14)
        self.assertEqual(res["info"]["prefill_calls"], 1)
        self.assertEqual(res["info"]["decode_calls"], 13)
        self.assertEqual(res["info"]["forward_calls"], 14)
        self.assertTrue(res["info"]["labels_on_prefill_only"])
        self.assertTrue(res["info"]["proprio_on_prefill_only"])
        self.assertTrue(res["info"]["pixels_on_prefill_only"])
        self.assertEqual(sum(1 for t in res["ids"] if 31744 <= t <= 31999), 14)
        # the prefill saw the full prompt + 4 vision tokens + 1 proprio token
        self.assertEqual(model.calls[0]["len"], len(PROMPT))
        self.assertEqual(res["info"]["prefill_input_len"], len(PROMPT))
        self.assertTrue(all(x == 1 for x in res["info"]["decode_input_lens"]))

    def test_logits_processor_forces_legal_window(self):
        model = make_model()
        model.bias_token = 0                     # token 0 would win without the constraint
        pixel, prop, projector = make_inputs()
        adapter = OG.R42OfficialGenerateAdapter(model, projector, device="cpu").install()
        res = adapter.generate(PROMPT, pixel, prop.numpy())
        self.assertTrue(all(31744 <= t <= 31999 for t in res["ids"]))
        self.assertTrue(all(t == 0 for t in res["free_argmax_ids"]))

    def test_new_request_builds_a_new_cache(self):
        model = make_model()
        pixel, prop, projector = make_inputs()
        adapter = OG.R42OfficialGenerateAdapter(model, projector, device="cpu").install()
        r1 = adapter.generate(PROMPT, pixel, prop.numpy())
        n_first = len(model.cache_ids)
        r2 = adapter.generate(PROMPT, pixel, prop.numpy() * 0.0 - 0.5)
        self.assertEqual(len(model.cache_ids) - n_first, 1 + 13)
        self.assertNotEqual(id(model.cache_ids[0]), id(model.cache_ids[n_first]))
        self.assertEqual(r1["ids"], r1["ids"])   # deterministic greedy decode
        self.assertEqual(r1["info"]["prefill_calls"], 1)
        self.assertEqual(r2["info"]["prefill_calls"], 1)

    def test_wrong_token_count_fails_closed(self):
        model = make_model()
        pixel, prop, projector = make_inputs()
        adapter = OG.R42OfficialGenerateAdapter(model, projector, device="cpu").install()
        real = model.generate

        def short(*a, **k):
            out = real(*a, **k)
            out.sequences = out.sequences[:, : len(PROMPT) + 3]
            return out

        model.generate = short
        with self.assertRaises(OG.R42GenerateError):
            adapter.generate(PROMPT, pixel, prop.numpy())

    def test_missing_pixel_values_is_rejected(self):
        model = make_model()
        _pixel, prop, projector = make_inputs()
        adapter = OG.R42OfficialGenerateAdapter(model, projector, device="cpu").install()
        with self.assertRaises(OG.R42GenerateError):
            adapter.generate(PROMPT, None, prop.numpy())

    def test_uninstall_restores_the_stock_prepare(self):
        model = make_model()
        _pixel, _prop, projector = make_inputs()
        before = model.prepare_inputs_for_generation
        adapter = OG.R42OfficialGenerateAdapter(model, projector, device="cpu")
        adapter.install()
        self.assertIsNot(model.prepare_inputs_for_generation, before)
        adapter.uninstall()
        self.assertEqual(model.prepare_inputs_for_generation.__func__,
                         type(model).prepare_inputs_for_generation)


class TestPrepareInputsContract(unittest.TestCase):
    def test_decode_step_drops_stale_pixels(self):
        model = make_model()
        out = OG.prepare_inputs_for_generation_impl(
            model, input_ids=torch.zeros(1, 5, dtype=torch.long), past_key_values=object(),
            pixel_values=torch.zeros(1, 3, 8, 8))
        self.assertIsNone(out["pixel_values"])
        self.assertEqual(model._r43_pixel_drops, 1)

    def test_prefill_requires_pixels(self):
        model = make_model()
        with self.assertRaises(OG.R42GenerateError):
            OG.prepare_inputs_for_generation_impl(
                model, input_ids=torch.zeros(1, 5, dtype=torch.long), past_key_values=None)

    def test_prefill_keeps_proprio_and_labels(self):
        model = make_model()
        out = OG.prepare_inputs_for_generation_impl(
            model, input_ids=torch.zeros(1, 5, dtype=torch.long), past_key_values=None,
            pixel_values=torch.zeros(1, 3, 8, 8), attention_mask=torch.ones(1, 5),
            proprio="P", proprio_projector="J", labels="L", use_cache=True)
        self.assertEqual((out["proprio"], out["proprio_projector"], out["labels"]), ("P", "J", "L"))
        self.assertIsNone(out["past_key_values"])

    def test_decode_step_returns_only_the_new_token(self):
        model = make_model()
        ids = torch.arange(7).reshape(1, 7)
        out = OG.prepare_inputs_for_generation_impl(model, input_ids=ids,
                                                    past_key_values=object())
        self.assertEqual(out["input_ids"].shape, (1, 1))
        self.assertEqual(int(out["input_ids"][0, 0]), 6)
        self.assertIsNone(out["labels"])
        self.assertIsNone(out["pixel_values"])
        self.assertIsNone(out["proprio"])

    def test_inputs_embeds_is_rejected(self):
        model = make_model()
        with self.assertRaises(OG.R42GenerateError):
            OG.prepare_inputs_for_generation_impl(model, input_ids=None, past_key_values=None,
                                                  inputs_embeds=torch.zeros(1, 3, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
