### Test metrics (AP = average precision; BSS relative to best baseline chosen on Val)

| Target | Group | Model | N | Prevalence | AP | Lift | Brier | BSS (%) | F1_Test |
|---|---|---|---|---|---|---|---|---|---|
| onset_1 | ALL | LightGBM | 16052 | 0.1327 | 0.5489 | 4.137 | 0.082 | 3.85 | 0.5353 |
| onset_1 | ALL | Baseline_B | 16052 | 0.1327 | 0.5164 | 3.892 | 0.0853 | 0.0 | 0.4871 |
| onset_1 | ALL | Always_Positive | 16052 | 0.1327 | 0.1327 | 1.0 | 0.8673 | -916.58 | 0.2343 |
| onset_1 | Frontline | LightGBM | 629 | 0.7011 | 0.861 | 1.228 | 0.1967 | -5.62 | 0.8302 |
| onset_1 | Frontline | Baseline_B | 629 | 0.7011 | 0.8476 | 1.209 | 0.1862 | 0.0 | 0.8219 |
| onset_1 | Frontline | Always_Positive | 629 | 0.7011 | 0.7011 | 1.0 | 0.2989 | -60.51 | 0.8243 |
| onset_1 | Rear | LightGBM | 15423 | 0.1095 | 0.3808 | 3.477 | 0.0774 | 4.73 | 0.4769 |
| onset_1 | Rear | Baseline_B | 15423 | 0.1095 | 0.35 | 3.196 | 0.0812 | 0.0 | 0.4187 |
| onset_1 | Rear | Always_Positive | 15423 | 0.1095 | 0.1095 | 1.0 | 0.8905 | -996.64 | 0.1974 |
| onset_3 | ALL | LightGBM | 16052 | 0.2522 | 0.7689 | 3.049 | 0.1062 | 9.61 | 0.7019 |
| onset_3 | ALL | Baseline_B | 16052 | 0.2522 | 0.7169 | 2.843 | 0.1174 | 0.0 | 0.6581 |
| onset_3 | ALL | Always_Positive | 16052 | 0.2522 | 0.2522 | 1.0 | 0.7478 | -536.77 | 0.4028 |
| onset_3 | Frontline | LightGBM | 629 | 0.9141 | 0.9806 | 1.073 | 0.0741 | 3.47 | 0.9551 |
| onset_3 | Frontline | Baseline_B | 629 | 0.9141 | 0.9658 | 1.057 | 0.0767 | 0.0 | 0.9551 |
| onset_3 | Frontline | Always_Positive | 629 | 0.9141 | 0.9141 | 1.0 | 0.0859 | -11.88 | 0.9551 |
| onset_3 | Rear | LightGBM | 15423 | 0.2252 | 0.6886 | 3.058 | 0.1075 | 9.77 | 0.6615 |
| onset_3 | Rear | Baseline_B | 15423 | 0.2252 | 0.6245 | 2.773 | 0.1191 | 0.0 | 0.6117 |
| onset_3 | Rear | Always_Positive | 15423 | 0.2252 | 0.2252 | 1.0 | 0.7748 | -550.57 | 0.3676 |
| onset_6 | ALL | LightGBM | 16052 | 0.3353 | 0.8494 | 2.533 | 0.1117 | 12.99 | 0.7655 |
| onset_6 | ALL | Baseline_B | 16052 | 0.3353 | 0.7995 | 2.385 | 0.1284 | 0.0 | 0.7123 |
| onset_6 | ALL | Always_Positive | 16052 | 0.3353 | 0.3353 | 1.0 | 0.6647 | -417.72 | 0.5022 |
| onset_6 | Frontline | LightGBM | 629 | 0.965 | 0.9953 | 1.031 | 0.0332 | 15.42 | 0.9822 |
| onset_6 | Frontline | Baseline_D | 629 | 0.965 | 0.9836 | 1.019 | 0.0393 | 0.0 | 0.9822 |
| onset_6 | Frontline | Always_Positive | 629 | 0.965 | 0.965 | 1.0 | 0.035 | 11.0 | 0.9822 |
| onset_6 | Rear | LightGBM | 15423 | 0.3096 | 0.8101 | 2.617 | 0.1149 | 13.01 | 0.7367 |
| onset_6 | Rear | Baseline_B | 15423 | 0.3096 | 0.7512 | 2.426 | 0.1321 | 0.0 | 0.6799 |
| onset_6 | Rear | Always_Positive | 15423 | 0.3096 | 0.3096 | 1.0 | 0.6904 | -422.63 | 0.4728 |

### Block-bootstrap (by day) 95% CI of LightGBM − best baseline

| Target | Group | Baseline | AP_LGB_Mean | AP_LGB_CI_Low | AP_LGB_CI_High | AP_Diff_Mean | AP_Diff_CI_Low | AP_Diff_CI_High | Brier_Diff_Mean | Brier_Diff_CI_Low | Brier_Diff_CI_High |
|---|---|---|---|---|---|---|---|---|---|---|---|
| onset_1 | ALL | Baseline_B | 0.5495 | 0.5206 | 0.5757 | 0.0323 | 0.0068 | 0.0617 | -0.0032 | -0.0067 | -0.0004 |
| onset_1 | Frontline | Baseline_B | 0.8628 | 0.8269 | 0.8937 | 0.014 | -0.0097 | 0.039 | 0.0106 | 0.0015 | 0.0202 |
| onset_1 | Rear | Baseline_B | 0.3817 | 0.3533 | 0.4075 | 0.0317 | -0.0066 | 0.0726 | -0.0039 | -0.0072 | -0.001 |
| onset_3 | ALL | Baseline_B | 0.77 | 0.7364 | 0.7972 | 0.0513 | 0.0238 | 0.0833 | -0.0112 | -0.0192 | -0.0048 |
| onset_3 | Frontline | Baseline_B | 0.9803 | 0.9697 | 0.9892 | 0.0149 | 0.0044 | 0.0263 | -0.0026 | -0.0122 | 0.0058 |
| onset_3 | Rear | Baseline_B | 0.6881 | 0.6467 | 0.7258 | 0.0637 | 0.0223 | 0.1089 | -0.0117 | -0.0193 | -0.0047 |
| onset_6 | ALL | Baseline_B | 0.8505 | 0.8166 | 0.8805 | 0.049 | 0.0266 | 0.0781 | -0.0165 | -0.0253 | -0.0088 |
| onset_6 | Frontline | Baseline_D | 0.9952 | 0.9884 | 0.9992 | 0.0114 | 0.0026 | 0.0216 | -0.0062 | -0.0112 | -0.0007 |
| onset_6 | Rear | Baseline_B | 0.811 | 0.7718 | 0.8459 | 0.0579 | 0.0273 | 0.0979 | -0.0172 | -0.0266 | -0.0088 |
