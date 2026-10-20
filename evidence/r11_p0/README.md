# R11 P0 CPU evidence (GPU seconds = 0, optimizer steps = 0)

| file | what it is |
|---|---|
| red_regression.log | first run of tests/test_r11_p0_*.py against r11/p0-audit@8a444d1: 14 failed, 7 passed, 7 errors |
| green_regression.log | full suite after the fixes: 250 passed, 3 skipped |
| r10_2_r42_22of22.log | tests/test_r10_2_r42.py (imported from r10.2/42d-deploy-12task@e7cd9bb) still 22/22 |

Reproduce:
    /home/dgut/pusht_runtime/venv_oft/bin/python -m pytest tests -q
    /home/dgut/pusht_runtime/venv_oft/bin/python tests/test_r10_2_r42.py
