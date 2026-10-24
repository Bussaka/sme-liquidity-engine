"""Train the business-owner liquidity-stress model (owner side of the early warning system).

Usage:
    python train_owner_model.py [--data data/Train.csv] [--out models]

Evaluation uses customer-grouped CV (each customer appears in several rows) so the
scores reflect performance on businesses the model has never seen. Probabilities
are calibrated inside each training fold, never on the rows being scored.
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

from liquidity_engine.features import ScaleFreeFeatures, required_columns

TARGET = 'liquidity_stress_next_30d'
# Columns that identify one customer across their repeated rows
CUSTOMER_KEY = ['arpu', 'age', 'gender', 'region', 'segment', 'earning_pattern', 'x_90_d_activity_rate']
N_FOLDS = 5
RANDOM_STATE = 42
# Day 11 Optuna parameters (best single HistGB in eda.ipynb)
HGB_PARAMS = dict(learning_rate=0.03, max_iter=550, max_leaf_nodes=27, min_samples_leaf=24,
                  l2_regularization=4.58, random_state=RANDOM_STATE)
WATCH_RECALL = 0.80


def build_pipeline():
    model = CalibratedClassifierCV(HistGradientBoostingClassifier(**HGB_PARAMS), method='sigmoid', cv=3)
    return Pipeline([('features', ScaleFreeFeatures()), ('model', model)])


def grouped_oof(X, y, groups):
    oof = np.zeros(len(X))
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
        pipe = build_pipeline().fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        print(f'  fold {fold}: AUC {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.4f}', flush=True)
    return oof


def pick_thresholds(y, p):
    grid = np.round(np.arange(0.02, 0.91, 0.01), 2)
    f1s = [f1_score(y, p >= t) for t in grid]
    alert = float(grid[int(np.argmax(f1s))])
    # Highest threshold that still catches WATCH_RECALL of stressed businesses
    watch = float(max(t for t in grid if recall_score(y, p >= t) >= WATCH_RECALL))
    return watch, alert


def band_stats(y, p, t):
    pred = p >= t
    return {'threshold': t, 'precision': round(precision_score(y, pred), 4),
            'recall': round(recall_score(y, pred), 4), 'flag_rate': round(float(pred.mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/Train.csv')
    ap.add_argument('--out', default='models')
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X, y = df[required_columns()], df[TARGET]
    groups = df.groupby(CUSTOMER_KEY, sort=False).ngroup()
    print(f'{len(df)} rows, {groups.nunique()} customers, stress rate {y.mean():.3f}')

    print('Customer-grouped cross-validation...')
    oof = grouped_oof(X, y, groups)
    metrics = {'roc_auc': roc_auc_score(y, oof), 'log_loss': log_loss(y, oof),
               'pr_auc': average_precision_score(y, oof), 'brier': brier_score_loss(y, oof)}
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    watch, alert = pick_thresholds(y, oof)
    bands = {'watch': band_stats(y, oof, watch), 'alert': band_stats(y, oof, alert)}
    print(json.dumps({'cv': metrics, 'bands': bands}, indent=2))

    print('Fitting final pipeline on all rows...')
    pipe = build_pipeline().fit(X, y)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    joblib.dump(pipe, out / 'owner_liquidity_pipeline.joblib')
    metadata = {
        'model': 'owner_liquidity_stress',
        'target': f'{TARGET} (business-owner side)',
        'status': 'PROXY-TRAINED - did not transfer to Kenya Financial Diaries (AUC ~0.48, see '
                  'analysis/diaries_check.py); validate on Kenyan M-Pesa statements before use',
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'training_data': {'file': args.data, 'rows': len(df), 'customers': int(groups.nunique()),
                          'positive_rate': round(float(y.mean()), 4),
                          'note': 'Proxy mobile-money data (appears Tanzanian); features are currency-independent'},
        'cv': {'scheme': f'StratifiedGroupKFold({N_FOLDS}) by customer', **metrics},
        'risk_bands': bands,
        'input_columns': required_columns(),
        'feature_names': list(pipe.named_steps['features'].get_feature_names_out()),
        'hgb_params': HGB_PARAMS,
        'versions': {'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__},
    }
    (out / 'owner_liquidity_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved {out / "owner_liquidity_pipeline.joblib"} and metadata.')


if __name__ == '__main__':
    main()
