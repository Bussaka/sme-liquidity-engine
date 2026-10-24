# Kenyan pilot data

The owner and creditor models are trained on proxy data and have not been validated in Kenya
(see `models/*_metadata.json`). This pilot collects consented Kenyan data so that we can validate
them and then retrain them.

## What to collect, per business

| Item | Detail |
|---|---|
| M-Pesa statements | **12 months**, at least 7 (6 months of history + 30 days to observe the outcome). Personal statements (PDF) and/or M-PESA for Business Till/Paybill exports (CSV). Overlapping periods are fine; duplicate receipts are removed. |
| Loan records (from the lending partner) | One row per loan: `loan_id`, `disbursement_date`, `due_date`, `principal`, `days_past_due` (or `defaulted` 0/1). |
| Consent record | Reference and date of the signed consent (see below). Businesses without both are skipped automatically. |

**Sample size.** Aim for **100–150 businesses**. Each 12-month statement gives about 6 labelled
monthly snapshots. At an expected stress rate of 15–25%, that is enough positives for
`analysis/pilot_validation.py` to give meaningful AUCs. It refuses to report on fewer than 20.

## File layout (everything stays under `data/`, which is gitignored)

```
data/pilot/manifest.csv      business_id, statement_file, consent_reference, consent_date[, pdf_password]
data/pilot/statements/       statement files named in the manifest
data/pilot/loans.csv         business_id, loan_id, disbursement_date, due_date, principal, days_past_due
```

One business can have several manifest rows (one per statement file). `pdf_password` is only needed
for encrypted PDF statements; remove it from the manifest once the build has run.

## Running it

```
set PILOT_SALT=<long random secret, stored outside the repo>
python build_pilot_dataset.py
python analysis/pilot_validation.py
```

Outputs are in `data/pilot/processed/`:
- `owner_snapshots.csv`: one row per business per 30-day date, with model inputs, Kenyan signals,
  label and owner score.
- `loan_rows.csv`: one row per loan, with statement features as of the day before disbursement and
  the default label.
- `quality_report.csv`: per business, the months covered and the share of transaction value the rules
  could not classify.

**After the first real statements arrive**, check `unclassified_share` in the quality report.
The classification rules in `liquidity_engine/mpesa_statement.py` (`RULES`, `LENDERS`, `BANKS`,
`BETTING`) follow the standard Safaricom wording, but have only been tested on synthetic
statements. Tune them until the unclassified share is below about 5%.

## Labels (to agree with the partner before training)

- **Owner stress, next 30 days** (proposed, `liquidity_engine/statement_aggregates.py`): Fuliza used on
  ≥ 5 days, or end-of-day balance below one day of normal spending on ≥ 10 days, or new digital-loan
  borrowing ≥ 50% of a normal month's inflow.
- **Loan default**: `days_past_due ≥ 30`, or the partner's `defaulted` flag.

## Privacy and consent (Kenya Data Protection Act, 2019)

What the pipeline already does:
- Raw transaction descriptions are dropped after classification.
- Counterparty names and numbers are replaced by keyed hashes (`PILOT_SALT`).
- Business IDs are hashed in every output.
- No data is committed to git.

What the organisation needs in place before collecting data (confirm with a Kenyan
data-protection lawyer; this is not legal advice):
- [ ] **Explicit, informed consent** from each business owner. It must name the purposes (early-warning
      alerts to the owner, and credit-risk assessment for the lender), what data is used, who sees it,
      and how long it is kept.
- [ ] **Registration with the ODPC** (Office of the Data Protection Commissioner) as data controller
      and/or processor.
- [ ] **A Data Protection Impact Assessment (DPIA)**: profiling for credit decisions is high-risk
      processing.
- [ ] **A data-sharing agreement** with the lending partner, covering the loan records.
- [ ] **Human review of automated flags.** People have a right not to be subject to decisions based
      solely on automated processing (s.35), so alerts must reach a person before any credit action.
- [ ] **A retention and deletion schedule** for the raw statements once they are processed.
