### Test metrics (AP = average precision; BSS relative to best baseline chosen on Val)

| Target | Group | Model | N | Prevalence | AP | Lift | Brier | BSS (%) | Precision_Test | Recall_Test | F1_Test |
|---|---|---|---|---|---|---|---|---|---|---|---|
| onset_1 | ALL | LightGBM | 17277 | 0.2661 | 0.5463 | 2.053 | 0.1583 | 1.23 | 0.4406 | 0.7286 | 0.5491 |
| onset_1 | ALL | Baseline_B | 17277 | 0.2661 | 0.5254 | 1.974 | 0.1603 | 0.0 | 0.4278 | 0.7403 | 0.5423 |
| onset_1 | ALL | Always_Positive | 17277 | 0.2661 | 0.2661 | 1.0 | 0.7339 | -357.8 | 0.2661 | 1.0 | 0.4204 |
| onset_3 | ALL | LightGBM | 17277 | 0.5224 | 0.8334 | 1.595 | 0.1661 | 0.64 | 0.6996 | 0.8936 | 0.7848 |
| onset_3 | ALL | Baseline_B | 17277 | 0.5224 | 0.8254 | 1.58 | 0.1671 | 0.0 | 0.6875 | 0.9064 | 0.7819 |
| onset_3 | ALL | Always_Positive | 17277 | 0.5224 | 0.5224 | 1.0 | 0.4776 | -185.75 | 0.5224 | 1.0 | 0.6863 |
| onset_6 | ALL | LightGBM | 17277 | 0.6725 | 0.9379 | 1.395 | 0.1217 | 0.22 | 0.8526 | 0.9267 | 0.8881 |
| onset_6 | ALL | Baseline_B | 17277 | 0.6725 | 0.9308 | 1.384 | 0.122 | 0.0 | 0.8288 | 0.9476 | 0.8842 |
| onset_6 | ALL | Always_Positive | 17277 | 0.6725 | 0.6725 | 1.0 | 0.3275 | -168.48 | 0.6725 | 1.0 | 0.8042 |

### Block-bootstrap (by day) 95% CI of LightGBM − best baseline

| Target | Group | Baseline | N_days | N_boot_used | AP_LGB_Mean | AP_LGB_CI_Low | AP_LGB_CI_High | AP_Diff_Mean | AP_Diff_CI_Low | AP_Diff_CI_High | Brier_Diff_Mean | Brier_Diff_CI_Low | Brier_Diff_CI_High |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onset_1 | ALL | Baseline_B | 43 | 1000 | 0.5463 | 0.5253 | 0.5651 | 0.0208 | 0.0096 | 0.0318 | -0.0019 | -0.0033 | -0.0006 |
| onset_3 | ALL | Baseline_B | 43 | 1000 | 0.8331 | 0.8165 | 0.8492 | 0.0078 | 0.0005 | 0.0162 | -0.001 | -0.0048 | 0.0023 |
| onset_6 | ALL | Baseline_B | 43 | 1000 | 0.9376 | 0.9265 | 0.9474 | 0.007 | 0.0013 | 0.0139 | -0.0003 | -0.006 | 0.0043 |
