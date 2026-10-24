"""Tests for the pilot pipeline on synthetic statements (no real customer data)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from liquidity_engine.features import required_columns
from liquidity_engine.mpesa_statement import classify, parse_statement_text, prepare, unclassified_share
from liquidity_engine.statement_aggregates import (kenyan_signals, model_inputs, owner_stress_label,
                                                   snapshot_dates)

ROOT = Path(__file__).resolve().parents[1]
SALT = b'test-salt'


def synthetic_statement(days=270, stressed_tail=False, seed=0):
    """A small shop: daily customer receipts, weekly stock paybill, agent withdrawals, airtime."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp('2025-01-01 08:00')
    rows, balance, n = [], 5000.0, 0

    def add(day, hour, details, amount):
        nonlocal balance, n
        balance += amount
        n += 1
        rows.append({'receipt': f'T{n:09d}', 'time': start + pd.Timedelta(days=day, hours=hour),
                     'details': details, 'status': 'COMPLETED', 'amount': round(amount, 2),
                     'balance': round(balance, 2)})

    for d in range(days):
        tail = stressed_tail and d >= days - 30
        for k in range(rng.integers(2, 6)):
            add(d, 1 + k, f'Funds received from 2547000000{k:02d} - CUSTOMER {k}', rng.uniform(200, 900) * (0.3 if tail else 1))
        if d % 7 == 0:
            add(d, 9, 'Pay Bill Online to 400200 - WHOLESALER LTD Acc. 55', -rng.uniform(3000, 6000))
            add(d, 9.1, 'Pay Bill Charge', -23.0)
        if d % 3 == 0:
            add(d, 10, 'Customer Withdrawal At Agent Till 123456 - SHOP AGENT', -rng.uniform(500, 1500))
        if d % 10 == 0:
            add(d, 11, 'Buy Bundles Online', -100.0)
        if tail and d % 2 == 0:
            add(d, 12, 'OverDraft of Credit Party', 800.0)
            add(d, 12.1, 'Merchant Payment Online to 777888 - SUPPLIER SHOP', -900.0)
    return pd.DataFrame(rows)


def test_classify_kenyan_transactions():
    assert classify('Funds received from 254711111111 - JANE W', 500) == ('received', 'received')
    assert classify('Customer Transfer to 254722222222 - JOHN K', -300) == ('mm_send', 'mm_send')
    assert classify('Pay Bill Online to 888880 - KPLC PREPAID Acc. 1234', -1000) == ('paybill', 'paybill')
    assert classify('Merchant Payment Online to 123456 - NAIVAS', -650) == ('merchantpay', 'merchantpay')
    assert classify('Customer Withdrawal At Agent Till 98765 - AGENT', -2000) == ('withdraw', 'withdraw')
    assert classify('Deposit of Funds at Agent Till 98765 - AGENT', 3000) == ('deposit', 'deposit')
    assert classify('Business Payment from 303030 - EQUITY BANK via API', 10000) == ('transfer_from_bank', 'transfer_from_bank')
    assert classify('Business Payment from 303030 - EQUITY BANK via API', -10000)[0] == 'transfer_to_bank'
    assert classify('Business Payment from 222333 - TALA KENYA', 2500) == ('digital_loan_in', 'received')
    assert classify('Pay Bill Online to 851900 - TALA KENYA Acc. 07', -2700) == ('loan_repayment', 'paybill')
    assert classify('OverDraft of Credit Party', 400)[0] == 'fuliza_draw'
    assert classify('OD Loan Repayment to 232323 - M-PESA Overdraw', -410)[0] == 'fuliza_repay'
    assert classify('Customer Transfer of Funds Charge', -13)[0] == 'charge'
    # Description says inflow but money went out: surfaced, not guessed
    assert classify('Funds received from 254711111111 - JANE W', -500)[0] == 'unclassified'


def test_parse_statement_text_handles_wrapped_rows():
    text = '\n'.join([
        'M-PESA STATEMENT',
        'Receipt No. Completion Time Details Transaction Status Paid In Withdrawn Balance',
        'SAB1CD2EF3 2025-03-02 10:15:00 Funds received from 254711111111 - JANE COMPLETED 1,500.00 0.00 6,500.00',
        'SAB1CD2EF4 2025-03-02 12:00:00 Pay Bill Online to 888880 - KPLC',
        'PREPAID Acc. 1234 COMPLETED 0.00 -1,000.00 5,500.00',
        'SAB1CD2EF5 2025-03-03 09:00:00 Customer Transfer to 254722222222 - JOHN FAILED 0.00 -200.00 5,500.00',
    ])
    df = parse_statement_text(text)
    assert list(df['receipt']) == ['SAB1CD2EF3', 'SAB1CD2EF4', 'SAB1CD2EF5']
    assert df.loc[1, 'details'] == 'Pay Bill Online to 888880 - KPLC PREPAID Acc. 1234'
    assert df['amount'].tolist() == [1500.0, -1000.0, -200.0]
    assert df.loc[1, 'balance'] == 5500.0
    tx = prepare(df, SALT)
    assert len(tx) == 2  # failed transaction dropped
    assert 'details' not in tx.columns  # raw text never leaves the parser


def test_counterparties_are_pseudonymised():
    tx = prepare(synthetic_statement(days=20), SALT)
    keys = tx['counterparty'].dropna()
    assert keys.str.fullmatch(r'[0-9a-f]{16}').all()
    assert not prepare(synthetic_statement(days=20), b'other-salt')['counterparty'].dropna().isin(keys).any()


def test_model_inputs_match_owner_pipeline():
    tx = prepare(synthetic_statement(), SALT)
    assert unclassified_share(tx) == 0
    dates = snapshot_dates(tx)
    assert dates and dates[0] >= tx['time'].min().normalize() + pd.Timedelta(days=180)
    row = model_inputs(tx, dates[0])
    assert list(row) == required_columns()
    assert row['m1_received_volume'] > 0 and row['m1_daily_avg_bal'] > 0
    assert row['earning_pattern'] in {'Weekly Earner', 'Monthly Earner', 'Irregular Earner'}
    model_path = ROOT / 'models' / 'owner_liquidity_pipeline.joblib'
    if model_path.exists():
        import joblib
        p = joblib.load(model_path).predict_proba(pd.DataFrame([row]))[:, 1]
        assert 0 <= p[0] <= 1


def test_stress_label_and_signals():
    calm = prepare(synthetic_statement(seed=1), SALT)
    stressed = prepare(synthetic_statement(seed=1, stressed_tail=True), SALT)
    as_of = calm['time'].max().normalize() - pd.Timedelta(days=30)
    assert owner_stress_label(calm, as_of) == 0
    assert owner_stress_label(stressed, as_of) == 1
    assert owner_stress_label(calm, calm['time'].max()) is None  # no future window yet
    sig = kenyan_signals(stressed, stressed['time'].max())
    assert sig['fuliza_days_30d'] >= 10 and sig['fuliza_share_of_inflow_90d'] > 0


def test_build_pilot_dataset_end_to_end(tmp_path):
    pilot = tmp_path / 'pilot'
    (pilot / 'statements').mkdir(parents=True)
    for i, stressed in enumerate([False, True]):
        df = synthetic_statement(seed=i, stressed_tail=stressed)
        pd.DataFrame({'Receipt No.': df.receipt, 'Completion Time': df.time, 'Details': df.details,
                      'Transaction Status': 'Completed', 'Paid In': df.amount.clip(lower=0),
                      'Withdrawn': (-df.amount).clip(lower=0), 'Balance': df.balance}
                     ).to_csv(pilot / 'statements' / f'biz{i}.csv', index=False)
    pd.DataFrame({'business_id': ['B0', 'B1', 'B2'], 'statement_file': ['biz0.csv', 'biz1.csv', 'biz0.csv'],
                  'consent_reference': ['C-001', 'C-002', None], 'consent_date': ['2025-09-01', '2025-09-01', None]}
                 ).to_csv(pilot / 'manifest.csv', index=False)
    pd.DataFrame({'business_id': ['B1'], 'loan_id': ['L1'], 'disbursement_date': ['2025-08-15'],
                  'due_date': ['2025-09-14'], 'principal': [5000], 'days_past_due': [45]}
                 ).to_csv(pilot / 'loans.csv', index=False)
    run = subprocess.run([sys.executable, 'build_pilot_dataset.py', '--pilot', str(pilot)], cwd=ROOT,
                         capture_output=True, text=True, env={**__import__('os').environ, 'PILOT_SALT': 'x'})
    assert run.returncode == 0, run.stderr
    assert 'Skipping 1 business' in run.stdout
    snaps = pd.read_csv(pilot / 'processed' / 'owner_snapshots.csv')
    assert snaps['business_key'].nunique() == 2 and 'B0' not in snaps['business_key'].values
    loans = pd.read_csv(pilot / 'processed' / 'loan_rows.csv')
    assert loans.loc[0, 'defaulted'] == 1 and bool(loans.loc[0, 'full_history'])
    quality = pd.read_csv(pilot / 'processed' / 'quality_report.csv')
    assert (quality['unclassified_share'] == 0).all()
