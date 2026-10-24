# SME Liquidity Engine

A dual-sided early warning system for Kenyan small businesses. It reads a business's M-Pesa cash
flow and flags liquidity stress before it turns into a missed payment. That protects the **business
owner** from running out of cash and the **creditor** from defaults.

> **Status: not yet validated in Kenya.** Both models are trained on public or proxy data and carry
> a `status` warning in their metadata. Neither should drive credit decisions until it has been
> validated on the Kenyan pilot (see [pilot/README.md](pilot/README.md)).

## Models

| Model | Predicts | Trained on | Result | Status |
|---|---|---|---|---|
| Owner (`models/owner_liquidity_pipeline.joblib`) | Liquidity stress in the next 30 days | Mobile-money proxy data (40,000 rows, 10,000 customers; appears Tanzanian) | AUC 0.916 on unseen customers | Did **not** transfer to Kenyan household data (AUC about 0.48 on the Kenya Financial Diaries) |
| Creditor (`models/creditor_default_pipeline.joblib`) | Loan default | Zindi African Credit Scoring, Kenyan digital loans (68,654 loans) | AUC 0.596 on later loans (out-of-time) | Baseline only: the data has no cash-flow or repayment behaviour |

**Owner model:**
- **Currency-independent features.** All 239 features are counts, shares, trends, cash runway and
  similar ratios, none tied to TZS or KES. They are computed inside the pipeline from raw monthly
  M-Pesa aggregates.
- **Honest validation.** Cross-validation is grouped by customer, so every score is on customers the
  model never saw. Probabilities are calibrated inside each training fold.
- **Two risk levels:**
  - **Watch:** threshold 0.18; catches 80% of stressed businesses (recall), with 50% precision.
  - **Alert:** threshold 0.35; recall 65%, precision 65%.

**Creditor model:**
- **History without leakage.** Borrower-history features use only earlier loans: loan stacking,
  borrowing frequency, amount growth, and past defaults counted once their due date has passed.
- **One column excluded.** `Total_Amount_to_Repay` is left out on purpose. It includes penalties
  added after default, which is target leakage.

Metrics, thresholds, input columns and library versions for each model are in
`models/*_metadata.json`.

## Repository layout

```
liquidity_engine/
  features.py              owner-model features (scale-free, from monthly M-Pesa aggregates)
  credit_features.py       creditor-model features (borrower loan history, Kenyan calendar effects)
  mpesa_statement.py       parse and classify M-Pesa statements (PDF / M-PESA for Business CSV), pseudonymised
  statement_aggregates.py  statements -> model inputs, Kenyan signals (Fuliza, digital loans, betting), stress labels
train_owner_model.py       train + save the owner pipeline
train_creditor_model.py    train + save the creditor pipeline
build_pilot_dataset.py     consented statements + loan book -> labelled pilot dataset
analysis/
  diaries_check.py         owner features vs Kenya Financial Diaries
  pilot_validation.py      validate both models on the pilot dataset
pilot/README.md            pilot data contract, labels, consent and Data Protection Act checklist
tests/                     pipeline tests on synthetic statements
models/                    saved pipelines + metadata
eda.ipynb                  original competition notebook (exploration history)
api.py, liquidity_risk_engine.pkl   legacy API and model, to be replaced
```

## Setup

```
pip install -r requirements.txt
python -m pytest tests
```

## Training

```
python train_owner_model.py        # needs data/Train.csv
python train_creditor_model.py     # needs data/external/zindi_credit/Train.csv
```

The owner run takes about 10 minutes. Both scripts print their evaluation, then save a pipeline
and metadata in `models/`. A pipeline takes raw columns, so no preprocessing is needed before scoring:

```python
import joblib, pandas as pd
owner = joblib.load('models/owner_liquidity_pipeline.joblib')
risk = owner.predict_proba(pd.read_csv('data/Test.csv'))[:, 1]
```

## Kenyan pilot

```
set PILOT_SALT=<secret kept outside the repo>
python build_pilot_dataset.py
python analysis/pilot_validation.py
```

See [pilot/README.md](pilot/README.md) for the file layout, the proposed stress label, and the consent
and Kenya Data Protection Act requirements. Before collecting data, a Kenyan data-protection lawyer
should confirm the approach.

## Data sources

All data lives under `data/`, which is gitignored and never committed.

| Source | Used for | Terms |
|---|---|---|
| Mobile-money proxy (`data/Train.csv`, `Test.csv`) | Owner-model training | Competition data |
| [Zindi African Credit Scoring](https://zindi.africa/competitions/african-credit-scoring-challenge) | Creditor-model training | Check Zindi's rules before any commercial use |
| [Kenya Financial Diaries 2012–13](https://doi.org/10.7910/DVN/JF8YST), FSD Kenya | Feature validation only | **Research only**; no redistribution, not for commercial use |
| [FinAccess 2021](https://statistics.knbs.or.ke/nada/index.php/catalog/104), KNBS/CBK/FSD Kenya | National default baselines | KNBS terms of use |
| World Bank Enterprise Survey Kenya | SME baselines (pending access) | World Bank Microdata terms |

## Legacy API

`api.py` serves the original competition model (`liquidity_risk_engine.pkl`). That model is the
uncalibrated Day 15 notebook model, and it expects pre-encoded categories and engineered features.
The API will be rebuilt around the new pipelines once the models are validated on pilot data.
