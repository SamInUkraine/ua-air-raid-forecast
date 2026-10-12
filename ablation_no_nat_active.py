"""Ablation: the main model without the national feature `nat_active_oblasts_cnt`.

That feature uses the 'is active at t' state of other oblasts (started_at <= t < finished_at), which is causal
(an alert still running at t is known to be active), but it is the only feature whose availability in real time
depends on how the source records end times. This run checks that the results do not rely on it.
Usage: python ablation_no_nat_active.py [--data PATH] [--results_dir results_ablation]
"""
import sys
import run_pipeline as rp

rp.FEATURES = [f for f in rp.FEATURES if f != 'nat_active_oblasts_cnt']
if '--results_dir' not in sys.argv:
    sys.argv += ['--results_dir', 'results_ablation']
rp.main()
