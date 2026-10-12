### Sensitivity runs (test period; BSS = 1 - Brier_LGB / Brier_best-baseline-on-Val)

| Run | Test (UTC) | Target | Group | N | Prevalence | Baseline | AP LGB | AP base | Brier LGB | Brier base | BSS % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A  official, main (causal ends, open-alert snapshot) | 2026-07-27 00 .. 2026-09-07 00 | onset_1 | ALL | 15984 | 0.1353 | Baseline_B | 0.5427 | 0.5361 | 0.084 | 0.0841 | 0.11 |
| A  official, main (causal ends, open-alert snapshot) | 2026-07-27 00 .. 2026-09-07 00 | onset_1 | Rear | 15351 | 0.1119 | Baseline_B | 0.3744 | 0.3792 | 0.0794 | 0.0799 | 0.6 |
| A  official, main (causal ends, open-alert snapshot) | 2026-07-27 00 .. 2026-09-07 00 | onset_3 | ALL | 15984 | 0.2574 | Baseline_B | 0.7723 | 0.7494 | 0.1061 | 0.1099 | 3.5 |
| A  official, main (causal ends, open-alert snapshot) | 2026-07-27 00 .. 2026-09-07 00 | onset_3 | Rear | 15351 | 0.2303 | Baseline_B | 0.6907 | 0.675 | 0.1075 | 0.1113 | 3.36 |
| A  official, main (causal ends, open-alert snapshot) | 2026-07-27 00 .. 2026-09-07 00 | onset_6 | ALL | 15984 | 0.3428 | Baseline_B | 0.8565 | 0.8343 | 0.1095 | 0.1162 | 5.75 |
| A  official, main (causal ends, open-alert snapshot) | 2026-07-27 00 .. 2026-09-07 00 | onset_6 | Rear | 15351 | 0.3172 | Baseline_B | 0.8176 | 0.7987 | 0.1126 | 0.1193 | 5.6 |
| A0 official, as-recorded fake ends (pre-audit data semantics) | 2026-07-27 00 .. 2026-09-07 00 | onset_1 | ALL | 16484 | 0.1325 | Baseline_B | 0.5373 | 0.5129 | 0.0826 | 0.0856 | 3.52 |
| A0 official, as-recorded fake ends (pre-audit data semantics) | 2026-07-27 00 .. 2026-09-07 00 | onset_1 | Rear | 15834 | 0.1094 | Baseline_B | 0.374 | 0.348 | 0.0778 | 0.0813 | 4.32 |
| A0 official, as-recorded fake ends (pre-audit data semantics) | 2026-07-27 00 .. 2026-09-07 00 | onset_3 | ALL | 16484 | 0.2524 | Baseline_B | 0.7572 | 0.7149 | 0.1081 | 0.118 | 8.39 |
| A0 official, as-recorded fake ends (pre-audit data semantics) | 2026-07-27 00 .. 2026-09-07 00 | onset_3 | Rear | 15834 | 0.2253 | Baseline_B | 0.6721 | 0.6221 | 0.1094 | 0.1195 | 8.53 |
| A0 official, as-recorded fake ends (pre-audit data semantics) | 2026-07-27 00 .. 2026-09-07 00 | onset_6 | ALL | 16484 | 0.3365 | Baseline_B | 0.8466 | 0.8015 | 0.1114 | 0.1277 | 12.82 |
| A0 official, as-recorded fake ends (pre-audit data semantics) | 2026-07-27 00 .. 2026-09-07 00 | onset_6 | Rear | 15834 | 0.3107 | Baseline_B | 0.8027 | 0.7533 | 0.1145 | 0.1314 | 12.84 |
| C  official, without nat_active_oblasts_cnt | 2026-07-27 00 .. 2026-09-07 00 | onset_1 | ALL | 15984 | 0.1353 | Baseline_B | 0.5552 | 0.5361 | 0.0838 | 0.0841 | 0.41 |
| C  official, without nat_active_oblasts_cnt | 2026-07-27 00 .. 2026-09-07 00 | onset_1 | Rear | 15351 | 0.1119 | Baseline_B | 0.3887 | 0.3792 | 0.0791 | 0.0799 | 1.0 |
| C  official, without nat_active_oblasts_cnt | 2026-07-27 00 .. 2026-09-07 00 | onset_3 | ALL | 15984 | 0.2574 | Baseline_B | 0.7803 | 0.7494 | 0.1046 | 0.1099 | 4.88 |
| C  official, without nat_active_oblasts_cnt | 2026-07-27 00 .. 2026-09-07 00 | onset_3 | Rear | 15351 | 0.2303 | Baseline_B | 0.7031 | 0.675 | 0.1059 | 0.1113 | 4.79 |
| C  official, without nat_active_oblasts_cnt | 2026-07-27 00 .. 2026-09-07 00 | onset_6 | ALL | 15984 | 0.3428 | Baseline_B | 0.8641 | 0.8343 | 0.1058 | 0.1162 | 8.89 |
| C  official, without nat_active_oblasts_cnt | 2026-07-27 00 .. 2026-09-07 00 | onset_6 | Rear | 15351 | 0.3172 | Baseline_B | 0.8269 | 0.7987 | 0.1088 | 0.1193 | 8.79 |
| V  volunteer, main (naive end := next start) | 2026-08-31 00 .. 2026-10-11 23 | onset_1 | ALL | 17277 | 0.2661 | Baseline_B | 0.5463 | 0.5254 | 0.1583 | 0.1603 | 1.23 |
| V  volunteer, main (naive end := next start) | 2026-08-31 00 .. 2026-10-11 23 | onset_3 | ALL | 17277 | 0.5224 | Baseline_B | 0.8334 | 0.8254 | 0.1661 | 0.1671 | 0.64 |
| V  volunteer, main (naive end := next start) | 2026-08-31 00 .. 2026-10-11 23 | onset_6 | ALL | 17277 | 0.6725 | Baseline_B | 0.9379 | 0.9308 | 0.1217 | 0.122 | 0.22 |
| V0 volunteer, naive fake 30-min ends kept | 2026-08-31 00 .. 2026-10-11 23 | onset_1 | ALL | 21210 | 0.2383 | Baseline_B | 0.5177 | 0.498 | 0.1492 | 0.1507 | 1.0 |
| V0 volunteer, naive fake 30-min ends kept | 2026-08-31 00 .. 2026-10-11 23 | onset_3 | ALL | 21210 | 0.4894 | Baseline_B | 0.801 | 0.7933 | 0.1767 | 0.1771 | 0.19 |
| V0 volunteer, naive fake 30-min ends kept | 2026-08-31 00 .. 2026-10-11 23 | onset_6 | ALL | 21210 | 0.6574 | Baseline_B | 0.9189 | 0.9141 | 0.1408 | 0.1359 | -3.59 |
| V1 volunteer, naive rows excluded | 2026-08-31 00 .. 2026-10-11 23 | onset_1 | ALL | 21461 | 0.2149 | Baseline_B | 0.5209 | 0.5009 | 0.1345 | 0.1362 | 1.26 |
| V1 volunteer, naive rows excluded | 2026-08-31 00 .. 2026-10-11 23 | onset_3 | ALL | 21461 | 0.4312 | Baseline_B | 0.7893 | 0.7796 | 0.1666 | 0.1688 | 1.32 |
| V1 volunteer, naive rows excluded | 2026-08-31 00 .. 2026-10-11 23 | onset_6 | ALL | 21461 | 0.5804 | Baseline_B | 0.8945 | 0.8877 | 0.1518 | 0.1529 | 0.72 |
