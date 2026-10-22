# R12 status (generated, do not hand-edit)

2->4 gate: **FAIL_ON_PHYSICAL_CONDITION** — the probability gate (P6) passes, but the PHYSICAL gate does not: on the frozen DEV frames the shared model's SAFE-14D error is worse than a zero-parameter constant-velocity extrapolator on both tasks, and its predicted residual has R^2 ~ 0. Per the pre-registered rule the 2->4 expansion stops and turns into mechanism diagnosis; the 2000-step checkpoint is preserved and no valid step is retrained.

| # | task | step_lim | expert | data | model SAFE L1 | no-op | CV | beats CV |
|---|---|---|---|---|---|---|---|---|
| 1 | stack_bowls | 800 | REALISTIC | PRESENT | 0.0070 | 0.0092 | 0.0022 | **no** |
| 2 | push_T | 600 | REALISTIC | PRESENT | 0.0014 | 0.0045 | 0.0009 | **no** |
| 3 | pack_objects_into_box | 1300 | REALISTIC | ABSENT | - | - | - | - |
| 4 | fold_clothes | 500 | BLOCKED_DATA | ABSENT | - | - | - | - |
| 5 | hang_mugs | 800 | HARD_BUT_RIGID | ABSENT | - | - | - | - |
| 6 | sweep_blocks | 1000 | HARD_BUT_RIGID | PRESENT | - | - | - | - |
| 7 | pour_liquid_into_cup | 400 | BLOCKED_DATA | ABSENT | - | - | - | - |
| 8 | make_toast | 1400 | HARD_BUT_RIGID | ABSENT | - | - | - | - |
| 9 | arrange_largest_number | 1050 | REALISTIC | ABSENT | - | - | - | - |
| 10 | sort_nesting_dolls_by_size | 1050 | REALISTIC | ABSENT | - | - | - | - |
| 11 | store_laptop_and_headphones | 800 | HARD_BUT_RIGID | ABSENT | - | - | - | - |
| 12 | stack_blocks | 550 | REALISTIC | ABSENT | - | - | - | - |

| # | task | expert | data | authorisation | next action |
|---|---|---|---|---|---|
| 1 | stack_bowls | 800 | REALISTIC | PRESENT (100 eps) | AUTHORISED | usable: TaskSpec verified, data present, included in the shared-model task pool |
| 2 | push_T | 600 | REALISTIC | PRESENT (100 eps) | AUTHORISED | usable: TaskSpec verified, data present, included in the shared-model task pool |
| 3 | pack_objects_into_box | 1300 | REALISTIC | ABSENT | AUTHORISED_FOR_PILOT | collect a bounded scripted-expert pilot under COLLECT harness (lanes/COLLECT_harness.md); gate P5_EXPERT_VALID |
| 4 | fold_clothes | 500 | BLOCKED_DATA | ABSENT | NOT_AUTHORISED | BLOCKED_DATA - rigid-body expert design cannot represent cloth/liquid; needs a physics-capable expert or moto/ |
| 5 | hang_mugs | 800 | HARD_BUT_RIGID | ABSENT | AUTHORISED_FOR_PILOT | collect a bounded scripted-expert pilot under COLLECT harness (lanes/COLLECT_harness.md); gate P5_EXPERT_VALID |
| 6 | sweep_blocks | 1000 | HARD_BUT_RIGID | PRESENT (100 eps) | AUTHORISED | usable: TaskSpec verified, data present, included in the shared-model task pool |
| 7 | pour_liquid_into_cup | 400 | BLOCKED_DATA | ABSENT | NOT_AUTHORISED | BLOCKED_DATA - rigid-body expert design cannot represent cloth/liquid; needs a physics-capable expert or moto/ |
| 8 | make_toast | 1400 | HARD_BUT_RIGID | ABSENT | AUTHORISED_FOR_PILOT | collect a bounded scripted-expert pilot under COLLECT harness (lanes/COLLECT_harness.md); gate P5_EXPERT_VALID |
| 9 | arrange_largest_number | 1050 | REALISTIC | ABSENT | AUTHORISED_FOR_PILOT | collect a bounded scripted-expert pilot under COLLECT harness (lanes/COLLECT_harness.md); gate P5_EXPERT_VALID |
| 10 | sort_nesting_dolls_by_size | 1050 | REALISTIC | ABSENT | AUTHORISED_FOR_PILOT | collect a bounded scripted-expert pilot under COLLECT harness (lanes/COLLECT_harness.md); gate P5_EXPERT_VALID |
| 11 | store_laptop_and_headphones | 800 | HARD_BUT_RIGID | ABSENT | AUTHORISED_FOR_PILOT | collect a bounded scripted-expert pilot under COLLECT harness (lanes/COLLECT_harness.md); gate P5_EXPERT_VALID |
| 12 | stack_blocks | 550 | REALISTIC | ABSENT | AUTHORISED_FOR_PILOT | collect a bounded scripted-expert pilot under COLLECT harness (lanes/COLLECT_harness.md); gate P5_EXPERT_VALID |

Counts: {"n_tasks": 12, "data_present": 3, "data_absent": 9, "blocked_data": 2}

2->4 gate: FAIL_ON_PHYSICAL_CONDITION

the probability gate (P6) passes, but the PHYSICAL gate does not: on the frozen DEV frames the shared model's SAFE-14D error is worse than a zero-parameter constant-velocity extrapolator on both tasks, and its predicted residual has R^2 ~ 0. Per the pre-registered rule the 2->4 expansion stops and turns into mechanism diagnosis; the 2000-step checkpoint is preserved and no valid step is retrained.
