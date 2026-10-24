"""Check the owner-model features against real Kenyan cash flows (Kenya Financial Diaries 2012-13).

Usage:
    python analysis/diaries_check.py

The Diaries have no stress label, so distress in month t+1 is defined from behaviour:
the household takes distress credit (friends/family, moneylender, chama, supplier,
store credit, M-Shwari) worth at least DISTRESS_SHARE of its median monthly spending.
Features come from months t-5..t only.

Data terms (FSD Kenya): research use only; outputs here are aggregates, never households.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from liquidity_engine.features import scale_free_features  # noqa: E402

DIARIES = 'data/external/kenya_financial_diaries/diaries_transactions_all.csv'
PROXY = 'data/Train.csv'
DISTRESS_TYPES = ['Friends and family: Borrowing', 'Moneylender or Shylock Borrowing',
                  'Borrowing from an informal Group', 'Supplier credit', 'Informal credit at a store',
                  'M-SHWARI Loan']
DISTRESS_SHARE = 0.10
WINDOW = 6
# Owner-model features that only need monthly inflow/outflow totals
TRANSFERABLE = ([f'm{i}_in_out_ratio' for i in range(1, 7)] + [f'm{i}_net_flow_share' for i in range(1, 7)] +
                ['inflow_recent_vs_prior', 'outflow_recent_vs_prior', 'in_out_recent', 'in_out_prior',
                 'neg_net_months', 'inflow_cv'])


def household_months():
    cols = ['hh_ids', 'trx_date', 'trx_class_desc', 'trx_family_desc', 'trx_type_desc', 'trx_mode_desc',
            'trx_direction', 'trx_value_kes']
    d = pd.read_csv(DIARIES, usecols=cols, low_memory=False)
    d = d[d['trx_direction'].isin(['Inflow', 'Outflow']) & (d['trx_mode_desc'] != 'In-kind (trade, good, service--NOT MONEY)')]
    d['month'] = pd.to_datetime(d['trx_date'], format='%d%b%Y').dt.to_period('M')
    v = d['trx_value_kes'].fillna(0)
    income = d['trx_class_desc'].str.startswith('Income')
    spend = d['trx_class_desc'].str.startswith('Operational')
    d['inflow'] = np.where(income & (d['trx_direction'] == 'Inflow'), v, 0)
    d['outflow'] = np.where((income | spend) & (d['trx_direction'] == 'Outflow'), v, 0)
    d['distress_credit'] = np.where(d['trx_type_desc'].isin(DISTRESS_TYPES) & (d['trx_direction'] == 'Inflow'), v, 0)
    d['business_income'] = np.where(income & (d['trx_family_desc'] == 'Self-employment') & (d['trx_direction'] == 'Inflow'), v, 0)
    d['mpesa_flow'] = np.where(d['trx_mode_desc'].str.startswith('Mobile money', na=False), v, 0)
    d['all_flow'] = v
    hm = d.groupby(['hh_ids', 'month'])[['inflow', 'outflow', 'distress_credit', 'business_income',
                                         'mpesa_flow', 'all_flow']].sum().reset_index()
    # First and last observed months are partial
    first_last = hm.groupby('hh_ids')['month'].transform
    return hm[(hm['month'] > first_last('min')) & (hm['month'] < first_last('max'))]


def windows(hm):
    rows = []
    for hh, g in hm.sort_values('month').groupby('hh_ids'):
        g = g.set_index('month')
        months = g.index
        median_spend = g['outflow'].median()
        for k in range(WINDOW, len(g)):
            hist = g.iloc[k - WINDOW:k]
            # Skip windows with a missing month
            if (months[k] - months[k - WINDOW]).n != WINDOW:
                continue
            nxt = g.iloc[k]
            row = {'hh': hh, 'target_month': months[k],
                   'distress_next': int(nxt['distress_credit'] >= DISTRESS_SHARE * max(median_spend, 1)),
                   'business_share': hist['business_income'].sum() / max(hist['inflow'].sum(), 1)}
            # Month 1 = most recent, matching the owner model's convention
            for i in range(1, 7):
                m = hist.iloc[WINDOW - i]
                row[f'm{i}_in'] = m['inflow']
                row[f'm{i}_out'] = m['outflow']
            rows.append(row)
    return pd.DataFrame(rows)


def transferable_features(inflow, outflow):
    """Same formulas as liquidity_engine.features for the totals-only subset."""
    total = inflow + outflow
    r = lambda a, b: (a + 1.0) / (b + 1.0)
    out = {}
    for i in range(1, 7):
        out[f'm{i}_in_out_ratio'] = r(inflow[:, i - 1], outflow[:, i - 1])
        out[f'm{i}_net_flow_share'] = (inflow[:, i - 1] - outflow[:, i - 1]) / (total[:, i - 1] + 1)
    recent, prior = slice(0, 3), slice(3, 6)
    out['inflow_recent_vs_prior'] = r(inflow[:, recent].sum(1), inflow[:, prior].sum(1))
    out['outflow_recent_vs_prior'] = r(outflow[:, recent].sum(1), outflow[:, prior].sum(1))
    out['in_out_recent'] = r(inflow[:, recent].sum(1), outflow[:, recent].sum(1))
    out['in_out_prior'] = r(inflow[:, prior].sum(1), outflow[:, prior].sum(1))
    out['neg_net_months'] = (inflow < outflow).sum(1)
    out['inflow_cv'] = inflow.std(1) / (inflow.mean(1) + 1)
    return pd.DataFrame(out)[TRANSFERABLE]


def signed_auc(y, x):
    """AUC above 0.5 means higher values go with more stress."""
    return roc_auc_score(y, np.nan_to_num(x, nan=0.0))


def main():
    hm = household_months()
    mpesa_share = hm['mpesa_flow'].sum() / hm['all_flow'].sum()
    w = windows(hm)
    inflow = w[[f'm{i}_in' for i in range(1, 7)]].to_numpy(float)
    outflow = w[[f'm{i}_out' for i in range(1, 7)]].to_numpy(float)
    Fk = transferable_features(inflow, outflow)
    yk = w['distress_next']
    print(f'Diaries: {w.hh.nunique()} households, {len(w)} six-month windows, '
          f'distress next month {yk.mean():.1%}, M-Pesa share of money flows {mpesa_share:.1%}')

    proxy = pd.read_csv(PROXY)
    Fp = scale_free_features(proxy)[TRANSFERABLE]
    yp = proxy['liquidity_stress_next_30d']

    print('\n1) Feature direction: AUC vs stress (>0.5 = higher value, more stress)')
    print(f'{"feature":26s}{"proxy":>8s}{"Kenya":>8s}  agree')
    agree = 0
    for c in TRANSFERABLE:
        a_p, a_k = signed_auc(yp, Fp[c]), signed_auc(yk, Fk[c])
        same = (a_p - 0.5) * (a_k - 0.5) > 0
        agree += same
        print(f'{c:26s}{a_p:8.3f}{a_k:8.3f}  {"yes" if same else "NO"}')
    print(f'Same direction on {agree}/{len(TRANSFERABLE)} features')

    print('\n2) Transfer test: train on proxy (totals-only features), score Kenyan households')
    m = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, random_state=42).fit(Fp, yp)
    pk = m.predict_proba(Fk)[:, 1]
    print(f'Kenya AUC (all households): {roc_auc_score(yk, pk):.3f}')
    biz = w['business_share'] >= 0.3
    print(f'Kenya AUC (business households, >=30% income from self-employment, n={biz.sum()}): '
          f'{roc_auc_score(yk[biz], pk[biz]):.3f}')

    print('\n3) Seasonality: share of households in distress by calendar month')
    hm['median_spend'] = hm.groupby('hh_ids')['outflow'].transform('median').clip(lower=1)
    hm['distress'] = hm['distress_credit'] >= DISTRESS_SHARE * hm['median_spend']
    hm['net_share'] = (hm['inflow'] - hm['outflow']) / (hm['inflow'] + hm['outflow'] + 1)
    season = hm.groupby(hm['month'].dt.month).agg(distress_rate=('distress', 'mean'),
                                                   median_net_share=('net_share', 'median'),
                                                   households=('hh_ids', 'nunique'))
    print(season.round(3).to_string())


if __name__ == '__main__':
    main()
