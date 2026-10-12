# Historical (pre-audit) results — superseded, kept for transparency

Produced on 2026-10-12 by re-running the ORIGINAL pre-audit code (repo commit b78265a) on the same official data
(dataset commit d10c80f). Byte-identical to the results that were in the repo before the audit (verified with `cmp`).

They are NOT valid final results: the pre-audit data handling kept the source's fake 1-hour end times (retrospective
leakage into `is_active`, `hours_since_last_finish`, `nat_active_oblasts_cnt` and the set of forecast origins) and its
6-hour label cut-off ignored alerts still open at the end of the data. See the main README for the corrected results.
