"""Currency-independent (scale-free) features for the business-owner liquidity model.

Every feature is a count, share, ratio, trend or runway, so the model does not
depend on whether amounts are in TZS (training proxy) or KES (Kenyan target).
Inputs are the monthly mobile-money aggregates a consented M-Pesa statement can
provide; month 1 is assumed to be the most recent.

Mapping to M-Pesa statement transaction types:
    deposit            -> Cash deposit at agent
    withdraw           -> Customer withdrawal at agent
    received           -> Funds received (Send Money in)
    mm_send            -> Customer transfer (Send Money out)
    paybill            -> Pay Bill
    merchantpay        -> Buy Goods / Till
    transfer_from_bank -> Bank to M-Pesa transfer
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MONTHS = range(1, 7)
INFLOWS = ['deposit', 'received', 'transfer_from_bank']
OUTFLOWS = ['withdraw', 'mm_send', 'paybill', 'merchantpay']
FAMILIES = INFLOWS + OUTFLOWS
COUNTERPARTY = {'deposit': 'agents', 'withdraw': 'agents', 'received': 'senders', 'mm_send': 'recipients',
                'paybill': 'companies', 'merchantpay': 'merchants', 'transfer_from_bank': 'banks'}
EARNING_PATTERNS = {'Weekly Earner': 0, 'Monthly Earner': 1, 'Irregular Earner': 2}


def required_columns():
    """Raw input columns the transformer reads."""
    cols = ['x_90_d_activity_rate', 'earning_pattern']
    for i in MONTHS:
        for f in FAMILIES:
            cols += [f'm{i}_{f}_volume', f'm{i}_{f}_total_value', f'm{i}_{f}_highest_amount',
                     f'm{i}_{f}_{COUNTERPARTY[f]}']
        cols.append(f'm{i}_daily_avg_bal')
    return cols


def _ratio(a, b):
    return (a + 1.0) / (b + 1.0)


def scale_free_features(df):
    out = {}
    val = {f: np.column_stack([df[f'm{i}_{f}_total_value'] for i in MONTHS]).astype(float) for f in FAMILIES}
    vol = {f: np.column_stack([df[f'm{i}_{f}_volume'] for i in MONTHS]).astype(float) for f in FAMILIES}
    bal = np.column_stack([df[f'm{i}_daily_avg_bal'] for i in MONTHS]).astype(float)
    inflow = sum(val[f] for f in INFLOWS)
    outflow = sum(val[f] for f in OUTFLOWS)
    total = inflow + outflow
    recent, prior = slice(0, 3), slice(3, 6)

    for f in FAMILIES:
        for i in MONTHS:
            out[f'm{i}_{f}_volume'] = df[f'm{i}_{f}_volume'].to_numpy(float)
            out[f'm{i}_{f}_counterparties'] = df[f'm{i}_{f}_{COUNTERPARTY[f]}'].to_numpy(float)
            out[f'm{i}_{f}_flow_share'] = val[f][:, i - 1] / (total[:, i - 1] + 1)
            out[f'm{i}_{f}_peak_share'] = df[f'm{i}_{f}_highest_amount'].to_numpy(float) / (val[f][:, i - 1] + 1)
        out[f'{f}_value_recent_vs_prior'] = _ratio(val[f][:, recent].sum(1), val[f][:, prior].sum(1))
        out[f'{f}_volume_recent_vs_prior'] = _ratio(vol[f][:, recent].sum(1), vol[f][:, prior].sum(1))
        out[f'{f}_value_m1_vs_mean'] = _ratio(val[f][:, 0], val[f].mean(1))
        out[f'{f}_active_months'] = (vol[f] > 0).sum(1)

    bal_mean = bal.mean(1)
    for i in MONTHS:
        out[f'm{i}_bal_rel'] = bal[:, i - 1] / (bal_mean + 1)
        out[f'm{i}_in_out_ratio'] = _ratio(inflow[:, i - 1], outflow[:, i - 1])
        out[f'm{i}_net_flow_share'] = (inflow[:, i - 1] - outflow[:, i - 1]) / (total[:, i - 1] + 1)
        # Days of spending that the month's average balance would cover
        out[f'm{i}_runway_days'] = bal[:, i - 1] / (outflow[:, i - 1] / 30 + 1)
        out[f'm{i}_turnover'] = total[:, i - 1] / (bal[:, i - 1] + 1)

    runway = np.column_stack([out[f'm{i}_runway_days'] for i in MONTHS])
    out['bal_recent_vs_prior'] = _ratio(bal[:, recent].mean(1), bal[:, prior].mean(1))
    out['bal_cv'] = bal.std(1) / (bal_mean + 1)
    out['bal_min_rel'] = bal.min(1) / (bal_mean + 1)
    out['bal_log_slope'] = np.polyfit(np.arange(6), np.log1p(bal).T, 1)[0]
    out['inflow_recent_vs_prior'] = _ratio(inflow[:, recent].sum(1), inflow[:, prior].sum(1))
    out['outflow_recent_vs_prior'] = _ratio(outflow[:, recent].sum(1), outflow[:, prior].sum(1))
    out['in_out_recent'] = _ratio(inflow[:, recent].sum(1), outflow[:, recent].sum(1))
    out['in_out_prior'] = _ratio(inflow[:, prior].sum(1), outflow[:, prior].sum(1))
    out['runway_recent_vs_prior'] = _ratio(runway[:, recent].sum(1), runway[:, prior].sum(1))
    out['neg_net_months'] = (inflow < outflow).sum(1)
    out['inflow_cv'] = inflow.std(1) / (inflow.mean(1) + 1)
    out['activity_rate_90d'] = df['x_90_d_activity_rate'].to_numpy(float)
    out['earning_pattern'] = df['earning_pattern'].map(EARNING_PATTERNS).to_numpy(float)
    return pd.DataFrame(out, index=df.index)


class ScaleFreeFeatures(BaseEstimator, TransformerMixin):
    """Pipeline step: raw monthly aggregates -> scale-free feature matrix."""

    def fit(self, X, y=None):
        missing = [c for c in required_columns() if c not in X.columns]
        if missing:
            raise ValueError(f'Missing input columns: {missing[:10]}{"..." if len(missing) > 10 else ""}')
        return self

    def transform(self, X):
        feats = scale_free_features(X)
        self.feature_names_out_ = list(feats.columns)
        return feats

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
