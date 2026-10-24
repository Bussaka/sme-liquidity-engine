"""Build the Kenyan pilot dataset from consented M-Pesa statements and a partner loan book.

Usage:
    set PILOT_SALT=<secret kept outside the repo>
    python build_pilot_dataset.py [--pilot data/pilot]

Expected layout (all under data/, which is gitignored):
    data/pilot/manifest.csv     business_id, statement_file, consent_reference, consent_date[, pdf_password]
    data/pilot/statements/      the statement files named in the manifest (PDF or CSV export)
    data/pilot/loans.csv        optional: business_id, loan_id, disbursement_date, due_date, principal,
                                days_past_due (or defaulted)

Outputs in data/pilot/processed/:
    owner_snapshots.csv  one row per business per 30-day scoring date: model inputs, Kenyan
                         signals, owner_stress_next_30d label, owner model score
    loan_rows.csv        one row per loan: statement features as of disbursement + default label
    quality_report.csv   coverage and classification quality per business
See pilot/README.md for the data contract and consent requirements.
"""
import argparse
import hashlib
import hmac
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

from liquidity_engine.mpesa_statement import load_statement, merge_statements, unclassified_share
from liquidity_engine.statement_aggregates import (kenyan_signals, model_inputs, owner_stress_label,
                                                   snapshot_dates)

OWNER_MODEL = Path('models/owner_liquidity_pipeline.joblib')
DEFAULT_DPD = 30


def business_key(business_id, salt):
    return hmac.new(salt, str(business_id).encode(), hashlib.sha256).hexdigest()[:16]


def load_business(rows, statements_dir, salt):
    frames = []
    for r in rows.itertuples():
        password = getattr(r, 'pdf_password', None)
        frames.append(load_statement(statements_dir / r.statement_file,
                                     password=None if pd.isna(password) else str(password), salt=salt))
    return merge_statements(frames)


def feature_row(tx, as_of):
    return {**model_inputs(tx, as_of), **kenyan_signals(tx, as_of)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', default='data/pilot')
    args = ap.parse_args()
    salt = os.environ.get('PILOT_SALT', '').encode()
    if not salt:
        sys.exit('Set PILOT_SALT to a secret kept outside the repo before building the pilot dataset.')

    pilot = Path(args.pilot)
    manifest = pd.read_csv(pilot / 'manifest.csv')
    no_consent = manifest['consent_reference'].isna() | manifest['consent_date'].isna()
    if no_consent.any():
        skipped = sorted(manifest.loc[no_consent, 'business_id'].astype(str).unique())
        print(f'Skipping {len(skipped)} business(es) with no recorded consent: {skipped}')
        manifest = manifest[~no_consent]
    loans = pd.read_csv(pilot / 'loans.csv', parse_dates=['disbursement_date', 'due_date']) \
        if (pilot / 'loans.csv').exists() else None
    owner_model = joblib.load(OWNER_MODEL) if OWNER_MODEL.exists() else None

    snapshots, loan_rows, quality = [], [], []
    for business_id, rows in manifest.groupby('business_id'):
        key = business_key(business_id, salt)
        try:
            tx = load_business(rows, pilot / 'statements', salt)
        except Exception as e:  # one bad file should not stop the batch
            print(f'  {key}: could not load statements ({e})')
            quality.append({'business_key': key, 'error': str(e)})
            continue
        dates = snapshot_dates(tx)
        for as_of in dates:
            snapshots.append({'business_key': key, 'as_of': as_of.date(), **feature_row(tx, as_of),
                              'owner_stress_next_30d': owner_stress_label(tx, as_of)})
        if loans is not None:
            for loan in loans[loans['business_id'] == business_id].itertuples():
                as_of = loan.disbursement_date - pd.Timedelta(days=1)
                has_history = (as_of - tx['time'].min()).days >= 180
                dpd = getattr(loan, 'days_past_due', None)
                defaulted = getattr(loan, 'defaulted', None)
                label = int(dpd >= DEFAULT_DPD) if dpd is not None and not pd.isna(dpd) else defaulted
                loan_rows.append({'business_key': key, 'loan_id': loan.loan_id, 'as_of': as_of.date(),
                                  'principal': loan.principal,
                                  'term_days': (loan.due_date - loan.disbursement_date).days,
                                  'full_history': has_history,
                                  **(feature_row(tx, as_of) if has_history else {}), 'defaulted': label})
        quality.append({'business_key': key, 'transactions': len(tx),
                        'first_date': tx['time'].min().date(), 'last_date': tx['time'].max().date(),
                        'months_covered': round((tx['time'].max() - tx['time'].min()).days / 30, 1),
                        'snapshots': len(dates), 'unclassified_share': round(unclassified_share(tx), 4)})

    out = pilot / 'processed'
    out.mkdir(exist_ok=True)
    snap = pd.DataFrame(snapshots)
    if owner_model is not None and len(snap):
        snap['owner_score'] = owner_model.predict_proba(snap)[:, 1]
    snap.to_csv(out / 'owner_snapshots.csv', index=False)
    loans_df = pd.DataFrame(loan_rows)
    if owner_model is not None and len(loans_df) and loans_df['full_history'].any():
        full = loans_df['full_history']
        loans_df.loc[full, 'owner_score'] = owner_model.predict_proba(loans_df[full])[:, 1]
    loans_df.to_csv(out / 'loan_rows.csv', index=False)
    pd.DataFrame(quality).to_csv(out / 'quality_report.csv', index=False)

    labelled = snap['owner_stress_next_30d'].notna().sum() if len(snap) else 0
    print(f'{manifest.business_id.nunique()} businesses -> {len(snap)} snapshots ({labelled} labelled), '
          f'{len(loans_df)} loans. Written to {out}/')


if __name__ == '__main__':
    main()
