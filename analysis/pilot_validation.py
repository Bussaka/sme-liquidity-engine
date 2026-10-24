"""Validate the models on the Kenyan pilot dataset built by build_pilot_dataset.py.

Usage:
    python analysis/pilot_validation.py [--pilot data/pilot]

Reports how well the proxy-trained owner score and each Kenyan signal predict
(a) owner stress in the next 30 days and (b) loan default, with bootstrap intervals
resampled by business so repeated snapshots of one business are not over-counted.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SIGNALS = ['owner_score', 'fuliza_days_30d', 'fuliza_share_of_inflow_90d', 'digital_loan_share_of_inflow_90d',
           'loan_repayment_share_of_outflow_90d', 'betting_share_of_outflow_90d', 'low_balance_days_30d']
MIN_POSITIVES = 20
N_BOOT = 500


def grouped_auc(df, score, label, groups, rng):
    y, s = df[label].to_numpy(), df[score].to_numpy()
    point = roc_auc_score(y, s)
    keys = groups.unique()
    boots = []
    for _ in range(N_BOOT):
        idx =np.concatenate([np.flatnonzero(groups.to_numpy() == k) for k in rng.choice(keys, len(keys))])
        yb, sb = y[idx], s[idx]
        if yb.min() != yb.max():
            boots.append(roc_auc_score(yb, sb))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return point, lo, hi


def report(df, label, title, rng):
    df = df[df[label].notna()].reset_index(drop=True)
    pos = int(df[label].sum())
    print(f'\n{title}: {len(df)} rows, {df.business_key.nunique()} businesses, {pos} positives ({df[label].mean():.1%})')
    if pos < MIN_POSITIVES or pos == len(df):
        print(f'  Not enough positives yet (need {MIN_POSITIVES}); collect more pilot data before reading AUCs.')
        return
    for s in [c for c in SIGNALS if c in df]:
        auc, lo, hi = grouped_auc(df, s, label, df['business_key'], rng)
        print(f'  {s:38s} AUC {auc:.3f}  (95% CI {lo:.3f}-{hi:.3f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', default='data/pilot')
    args = ap.parse_args()
    out = Path(args.pilot) / 'processed'
    rng = np.random.default_rng(42)
    report(pd.read_csv(out / 'owner_snapshots.csv'), 'owner_stress_next_30d', 'Owner stress next 30 days', rng)
    loans = pd.read_csv(out / 'loan_rows.csv')
    if len(loans):
        report(loans[loans['full_history']], 'defaulted', 'Loan default', rng)


if __name__ == '__main__':
    main()
