"""Turn classified M-Pesa transactions into model inputs, Kenyan signals and stress labels.

Months are rolling 30-day windows ending at `as_of` (m1 = the most recent 30 days), so a
business can be scored on any date, not only at calendar month ends.

The owner-stress label below is a PROPOSED Kenyan definition, to be agreed with the
pilot partner before training on it. A business is in liquidity stress in the 30 days
after `as_of` if any of these happen:
    - Fuliza overdraft used on at least FULIZA_DAYS days
    - end-of-day balance below one day of normal spending on at least LOW_BALANCE_DAYS days
    - new digital-loan borrowing worth at least DISTRESS_LOAN_SHARE of a normal month's inflow
"""
import numpy as np
import pandas as pd

from liquidity_engine.features import COUNTERPARTY, FAMILIES, required_columns

WINDOW_DAYS = 30
HISTORY_MONTHS = 6
FULIZA_DAYS = 5
LOW_BALANCE_DAYS = 10
DISTRESS_LOAN_SHARE = 0.5
# Allow statements to end a few days short of a full label window
COVERAGE_SLACK_DAYS = 3
INFLOW_FAMILIES = ['deposit', 'received', 'transfer_from_bank']


def daily_balances(tx, start, end):
    """End-of-day balance for each calendar day in [start, end]."""
    first = tx.iloc[0]
    opening = first['balance'] - first['amount']
    eod = tx.groupby(tx['time'].dt.normalize())['balance'].last()
    days = pd.date_range(start.normalize(), end.normalize(), freq='D')
    before = eod[eod.index < days[0]]
    seed = before.iloc[-1] if len(before) else opening
    return eod.reindex(days).ffill().fillna(seed)


def _window(tx, start, end):
    return tx[(tx['time'] > start) & (tx['time'] <= end)]


def _earning_pattern(inflows, as_of):
    """Weekly / Monthly / Irregular from the last 90 days of income-type inflows."""
    recent = inflows[inflows['time'] > as_of - pd.Timedelta(days=90)]
    if recent.empty:
        return 'Irregular Earner'
    weekly = recent.groupby(((as_of - recent['time']).dt.days // 7))['amount'].sum().reindex(range(13), fill_value=0)
    if (weekly > 0).sum() >= 10 and weekly.std() / max(weekly.mean(), 1) < 1.0:
        return 'Weekly Earner'
    by_month = recent.groupby((as_of - recent['time']).dt.days // WINDOW_DAYS)
    concentrated = by_month.apply(lambda g: g.groupby(g['time'].dt.date)['amount'].sum().max() / max(g['amount'].sum(), 1))
    if (concentrated >= 0.5).sum() >= 2:
        return 'Monthly Earner'
    return 'Irregular Earner'


def model_inputs(tx, as_of):
    """Raw columns the owner pipeline expects (liquidity_engine.features.required_columns)."""
    as_of = pd.Timestamp(as_of)
    row = {}
    for i in range(1, HISTORY_MONTHS + 1):
        end = as_of - pd.Timedelta(days=WINDOW_DAYS * (i - 1))
        win = _window(tx, end - pd.Timedelta(days=WINDOW_DAYS), end)
        for f in FAMILIES:
            g = win[win['family'] == f]
            value = g['amount'].abs()
            row[f'm{i}_{f}_volume'] = len(g)
            row[f'm{i}_{f}_total_value'] = float(value.sum())
            row[f'm{i}_{f}_highest_amount'] = float(value.max()) if len(g) else 0.0
            row[f'm{i}_{f}_{COUNTERPARTY[f]}'] = int(g['counterparty'].nunique())
        bal = daily_balances(tx, end - pd.Timedelta(days=WINDOW_DAYS - 1), end)
        row[f'm{i}_daily_avg_bal'] = float(bal.mean())
    last90 = _window(tx, as_of - pd.Timedelta(days=90), as_of)
    row['x_90_d_activity_rate'] = last90['time'].dt.normalize().nunique() / 90
    row['earning_pattern'] = _earning_pattern(tx[tx['family'].isin(INFLOW_FAMILIES)], as_of)
    return {c: row[c] for c in required_columns()}


def _normal_levels(tx, as_of):
    """Average daily spending and monthly inflow over the 90 days before as_of."""
    hist = _window(tx, as_of - pd.Timedelta(days=90), as_of)
    spend = hist.loc[hist['amount'] < 0, 'amount'].abs().sum() / 90
    inflow = hist.loc[hist['family'].isin(INFLOW_FAMILIES), 'amount'].sum() / 3
    return spend, inflow


def kenyan_signals(tx, as_of):
    """Extra Kenyan signals (not in the proxy-trained model); shares keep them currency-free."""
    as_of = pd.Timestamp(as_of)
    last30 = _window(tx, as_of - pd.Timedelta(days=30), as_of)
    last90 = _window(tx, as_of - pd.Timedelta(days=90), as_of)
    out_value = last90.loc[last90['amount'] < 0, 'amount'].abs().sum()
    in_value = last90.loc[last90['amount'] > 0, 'amount'].sum()
    share = lambda mask, total: float(last90.loc[mask, 'amount'].abs().sum() / max(total, 1))
    cat = last90['category']
    spend_per_day, _ = _normal_levels(tx, as_of)
    bal30 = daily_balances(tx, as_of - pd.Timedelta(days=29), as_of)
    return {
        'fuliza_days_30d': int(last30.loc[last30['category'] == 'fuliza_draw', 'time'].dt.normalize().nunique()),
        'fuliza_share_of_inflow_90d': share(cat == 'fuliza_draw', in_value),
        'digital_loan_share_of_inflow_90d': share(cat == 'digital_loan_in', in_value),
        'loan_repayment_share_of_outflow_90d': share(cat.isin(['loan_repayment', 'fuliza_repay']), out_value),
        'betting_share_of_outflow_90d': share(last90['is_betting'] & (last90['amount'] < 0), out_value),
        'charges_share_of_outflow_90d': share(cat == 'charge', out_value),
        'savings_share_of_outflow_90d': share(cat == 'savings_out', out_value),
        'low_balance_days_30d': int((bal30 < spend_per_day).sum()),
        'unclassified_share_90d': share(cat == 'unclassified', in_value + out_value),
    }


def owner_stress_label(tx, as_of):
    """1/0 stress in the 30 days after as_of, or None when the statement does not cover them."""
    as_of = pd.Timestamp(as_of)
    end = as_of + pd.Timedelta(days=WINDOW_DAYS)
    if tx['time'].max() < end - pd.Timedelta(days=COVERAGE_SLACK_DAYS):
        return None
    future = _window(tx, as_of, end)
    spend_per_day, monthly_inflow = _normal_levels(tx, as_of)
    fuliza_days = future.loc[future['category'] == 'fuliza_draw', 'time'].dt.normalize().nunique()
    low_days = int((daily_balances(tx, as_of + pd.Timedelta(days=1), end) < spend_per_day).sum())
    new_loans = future.loc[future['category'] == 'digital_loan_in', 'amount'].sum()
    stressed = (fuliza_days >= FULIZA_DAYS or low_days >= LOW_BALANCE_DAYS
                or (monthly_inflow > 0 and new_loans >= DISTRESS_LOAN_SHARE * monthly_inflow))
    return int(stressed)


def snapshot_dates(tx):
    """Scoring dates every 30 days, each with 6 months of history behind it."""
    start = tx['time'].min().normalize() + pd.Timedelta(days=WINDOW_DAYS * HISTORY_MONTHS)
    last = tx['time'].max().normalize()
    return list(pd.date_range(start, last, freq=f'{WINDOW_DAYS}D'))
