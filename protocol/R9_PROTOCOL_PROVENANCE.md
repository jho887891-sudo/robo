# Protocol provenance (R9 Gate D)

The R9 evaluation workstream reuses the R8-C tooling/protocol unchanged in behaviour and
pins them by hash:

| file | source | sha256 |
|---|---|---|
| `tools/r8_eval_manifest.py` | `research/r8-independent-evaluation@4f28417c18747bc2b208650b690b444f4d7a8678` | `fccb4a5772428c77ee79a72c4e40318882056e50a66736e925fd472270711717` |
| `protocol/r8_eval_protocol.json` | same | `48ad7326f9ae938ecdd8d9e19e219297fcb594fbcbb823066cecc1163327aa55` |
| `tests/test_r8_eval_manifest.py` | same | `caa1069d63e565909c175e7f27b056e703a6b72a92b74aac8a2c598a8951fb56` |

`protocol/r9_eval_protocol.json` declares `extends.r8_protocol_sha256` and the regression test
fails if that hash no longer matches the committed file, so silent protocol drift is caught.
