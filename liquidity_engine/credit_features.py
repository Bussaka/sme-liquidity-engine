"""Features for the creditor-side default model (Kenyan digital loans).

Input is a loan table that includes each borrower's earlier loans. A loan's history
features use only loans disbursed strictly before it, and an earlier loan's outcome
(`defaulted`) only once its due date has passed, so nothing from the future leaks in.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

INPUT_COLUMNS = ['customer_id', 'tbl_loan_id', 'loan_type', 'Total_Amount', 'Total_Amount_to_Repay',
                 'disbursement_date', 'due_date', 'duration', 'New_versus_Repeat', 'Lender_portion_Funded']
OUTCOME = 'defaulted'
# Months when Kenyan school terms start and fee payments squeeze household cash
SCHOOL_FEE_MONTHS = {1, 5, 9}


def _history(loans):
    """Per-loan borrower history from strictly earlier loans (one row per tbl_loan_id)."""
    rows = []
    for _, g in loans.sort_values('disbursement_date').groupby('customer_id', sort=False):
        disb = g['disbursement_date'].to_numpy('datetime64[D]')
        due = g['due_date'].to_numpy('datetime64[D]')
        amt = g['Total_Amount'].to_numpy(float)
        outcome = g[OUTCOME].to_numpy(float) if OUTCOME in g else np.full(len(g), np.nan)
        for i in range(len(g)):
            prior = disb < disb[i]
            n = int(prior.sum())
            settled = prior & (due < disb[i]) & ~np.isnan(outcome)
            open_ = prior & (due >= disb[i])
            days_ago = (disb[i] - disb[prior]).astype(int)
            rows.append({
                'tbl_loan_id': g['tbl_loan_id'].iat[i],
                'prior_loans': n,
                'prior_loans_30d': int((days_ago <= 30).sum()),
                'prior_loans_90d': int((days_ago <= 90).sum()),
                'days_since_last_loan': days_ago.min() if n else np.nan,
                'days_since_first_loan': days_ago.max() if n else np.nan,
                'amount_vs_prior_mean': amt[i] / amt[prior].mean() if n else np.nan,
                'amount_vs_prior_max': amt[i] / amt[prior].max() if n else np.nan,
                'open_loans': int(open_.sum()),
                'open_amount_vs_amount': amt[open_].sum() / amt[i] if amt[i] > 0 else np.nan,
                'settled_loans': int(settled.sum()),
                'prior_defaults': float(np.nansum(outcome[settled])),
                'prior_default_rate': outcome[settled].mean() if settled.any() else np.nan,
            })
    return pd.DataFrame(rows)


def credit_features(df):
    df = df.copy()
    df['disbursement_date'] = pd.to_datetime(df['disbursement_date'])
    df['due_date'] = pd.to_datetime(df['due_date'])
    loans = df.drop_duplicates('tbl_loan_id')
    hist = _history(loans)
    out = df[['tbl_loan_id']].merge(hist, on='tbl_loan_id', how='left').drop(columns='tbl_loan_id')
    out.index = df.index

    amount = df['Total_Amount'].astype(float)
    out['log_amount'] = np.log1p(amount)
    # Total_Amount_to_Repay is deliberately unused: for identical 7-day Type_1 loans from one
    # lender, repaid loans carry a ~0.7% median markup and defaulted ones ~41%, so it includes
    # penalties accrued after default and is not known at disbursement.
    out['duration'] = df['duration']
    out['lender_share'] = df['Lender_portion_Funded']
    out['is_new_borrower'] = (df['New_versus_Repeat'] == 'New Loan').astype(int)
    out['loan_type'] = df['loan_type'].str.replace('Type_', '', regex=False).astype(float)
    out['disb_day_of_month'] = df['disbursement_date'].dt.day
    out['disb_weekday'] = df['disbursement_date'].dt.weekday
    out['due_day_of_month'] = df['due_date'].dt.day
    out['due_in_school_fee_month'] = df['due_date'].dt.month.isin(SCHOOL_FEE_MONTHS).astype(int)
    out['due_near_month_end'] = (df['due_date'].dt.day >= 25).astype(int)
    return out


class CreditFeatures(BaseEstimator, TransformerMixin):
    """Pipeline step: loan table (with borrower history) -> per-loan feature matrix."""

    def fit(self, X, y=None):
        missing = [c for c in INPUT_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f'Missing input columns: {missing}')
        return self

    def transform(self, X):
        feats = credit_features(X)
        self.feature_names_out_ = list(feats.columns)
        return feats

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
