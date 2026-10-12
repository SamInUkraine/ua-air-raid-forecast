"""Sensitivity: the main official model without `nat_active_oblasts_cnt`.

Equivalent to: python run_pipeline.py --drop-features nat_active_oblasts_cnt --results_dir results_ablation
nat_active_oblasts_cnt = number of oblasts with an alert active at t (🔴 seen, no 🟢 yet). It is real-time knowable,
but it is the feature most exposed to how the source records END times, so the conclusions are checked without it.
"""
import sys
import run_pipeline as rp

if '--drop-features' not in sys.argv:
    sys.argv += ['--drop-features', 'nat_active_oblasts_cnt']
if '--results_dir' not in sys.argv:
    sys.argv += ['--results_dir', 'results_ablation']
rp.main()
