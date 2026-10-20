# R10.2 C — failure / iteration log

| # | action | result | fix |
|---|---|---|---|
| 1 | first C3 patcher run | `SystemExit: mem_get_info marker not unique` (the string appeared in both `run_gpu_smoke` and `kv_diagnose`) | anchored on the unique `pre = kv_preflight(None, free, args.min_free_gb)` line instead |
| 2 | first C3 test-appender run | `anchor not unique` (the test file uses double quotes in `n = sum(... r["pass"])`) | corrected the anchor string; appended 4 C3 tests |
| 3 | C4 GPU run | **success on the first attempt**: exit 0, 98 s, 14/14 | - |

No gate criterion was relaxed; the GPU budget (1 run) was consumed exactly once and not retried.
