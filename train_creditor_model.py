"""Train the creditor-side loan default model (Kenyan digital loans, Zindi African Credit Scoring).

Usage:
    python train_creditor_model.py [--data data/external/zindi_credit/Train.csv] [--out models]

Headline evaluation is out-of-time: fit on loans disbursed before CUTOFF and score
the later loans, which is how a lender uses the model. Default rates shift strongly
over time in this data, so random CV alone would overstate performance.
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from liquidity_engine.credit_features import INPUT_COLUMNS, OUTCOME, CreditFeatures

CUTOFF = '2024-07-01'
N_FOLDS = 5
RANDOM_STATE = 42
HGB_PARAMS = dict(learning_rate=0.03, max_iter=400, max_leaf_nodes=15, min_samples_leaf=40,
                  l2_regularization=1.0, categorical_features=None, random_state=RANDOM_STATE)
WATCH_RECALL = 0.80


def build_pipeline():
    model = CalibratedClassifierCV(HistGradientBoostingClassifier(**HGB_PARAMS), method='sigmoid', cv=3)
    return Pipeline([('features', CreditFeatures()), ('model', model)])


def scores(y, p):
    return {k: round(float(v), 4) for k, v in {
        'roc_auc': roc_auc_score(y, p), 'pr_auc': average_precision_score(y, p),
        'log_loss': log_loss(y, p), 'brier': brier_score_loss(y, p),
        'default_rate': float(np.mean(y)), 'mean_predicted': float(np.mean(p))}.items()}


def pick_thresholds(y, p):
    # Quantile grid: predicted default probabilities are small and can sit far below 0.01
    grid = np.unique(np.round(np.quantile(p, np.linspace(0, 0.995, 400)), 5))
    alert = float(grid[int(np.argmax([f1_score(y, p >= t) for t in grid]))])
    watch = float(max(t for t in grid if recall_score(y, p >= t) >= WATCH_RECALL))
    return watch, alert


def band_stats(y, p, t):
    pred = p >= t
    return {'threshold': t, 'precision': round(precision_score(y, pred, zero_division=0), 4),
            'recall': round(recall_score(y, pred), 4), 'flag_rate': round(float(pred.mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/external/zindi_credit/Train.csv')
    ap.add_argument('--out', default='models')
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = df[df['country_id'] == 'Kenya'].reset_index(drop=True)
    df[OUTCOME] = df['target']
    X, y = df[INPUT_COLUMNS + [OUTCOME]], df['target']
    later = pd.to_datetime(df['disbursement_date']) >= CUTOFF
    print(f'{len(df)} loans, {df.customer_id.nunique()} borrowers, default rate {y.mean():.4f}')
    print(f'Out-of-time split at {CUTOFF}: {(~later).sum()} earlier / {later.sum()} later loans')

    # Score the later loans with the full frame so they see their borrowers' earlier history
    pipe = build_pipeline().fit(X[~later], y[~later])
    p_oot = pipe.predict_proba(X)[:, 1][later.to_numpy()]
    oot = scores(y[later], p_oot)
    watch, alert = pick_thresholds(y[later], p_oot)
    bands = {'watch': band_stats(y[later], p_oot, watch), 'alert': band_stats(y[later], p_oot, alert)}
    print('Out-of-time:', json.dumps({'scores': oot, 'bands': bands}, indent=2))

    print('Borrower-grouped CV (all periods mixed)...')
    oof = np.zeros(len(df))
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, va_idx in cv.split(X, y, df['customer_id']):
        fold_pipe = build_pipeline().fit(X.iloc[tr_idx], y.iloc[tr_idx])
        # History for held-out borrowers comes only from their own earlier loans
        oof[va_idx] = fold_pipe.predict_proba(X.iloc[va_idx])[:, 1]
    grouped = scores(y, oof)
    print('Grouped CV:', grouped)

    print('Fitting final pipeline on all Kenyan loans...')
    final = build_pipeline().fit(X, y)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    joblib.dump(final, out / 'creditor_default_pipeline.joblib')
    metadata = {
        'model': 'creditor_loan_default',
        'target': 'loan default (creditor side)',
        'status': 'BASELINE ONLY - weak out-of-time ranking; needs a lender loan book or repayment data',
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'training_data': {'file': args.data, 'loans': len(df), 'borrowers': int(df.customer_id.nunique()),
                          'default_rate': round(float(y.mean()), 4), 'country': 'Kenya',
                          'note': 'Zindi African Credit Scoring; research use only, check licence before commercial use'},
        'evaluation': {'out_of_time': {'cutoff': CUTOFF, **oot}, 'borrower_grouped_cv': grouped},
        'risk_bands': bands,
        'input_columns': INPUT_COLUMNS + [OUTCOME],
        'input_note': f'Include each borrower\'s earlier loans; `{OUTCOME}` is the known outcome of past loans (blank if unknown)',
        'feature_names': list(final.named_steps['features'].get_feature_names_out()),
        'hgb_params': {k: v for k, v in HGB_PARAMS.items() if k != 'categorical_features'},
        'versions': {'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__},
    }
    (out / 'creditor_default_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved {out / "creditor_default_pipeline.joblib"} and metadata.')


if __name__ == '__main__':
    main()
