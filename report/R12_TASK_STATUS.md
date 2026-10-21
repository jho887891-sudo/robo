# R12 status (generated, do not hand-edit)

| # | task | step_lim | expert | data | authorisation | next action |
|---|---|---|---|---|---|---|
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

2->4 gate: PERMITTED_PENDING_BUDGET_DECISION

P6_M2_TWO now PASSES on the corrected evidence (mission/m2_p6_two_corrected_report.json), so the pre-registered 2->4 condition is met. Starting a 4-task stage is a model/budget decision for the research lead and has NOT been started.
