# Freight Load Rate Prediction

Predicts `posted_rate` (USD) for freight loads. The final model is a **Ridge + LightGBM hybrid** trained on `train-test.csv` and used to score the 12,000 loads in `validation.csv` plus the 31 December chart rows.

## Repository layout

```
.
├── Model.ipynb                     # full pipeline: clean -> features -> compare models -> predict
├── score.py                        # provided scorer
├── requirements.txt
├── data/
│   ├── train-test.csv              # labeled development data (input)
│   ├── validation.csv              # loads to predict (input)
│   ├── validation-predictions-template.csv  (input)
│   ├── december-chart-inputs.csv   # chart rows without predictions (input)
│   
└── outputs/                       
    └── ridge_alpha_comparison.csv  # model-comparison tables written by the notebook
    └── ridge_vs_hybrid.csv  # model-comparison tables written by the notebook
    └── december_chart_inputs.csv   # FINAL OUTPUT: chart rows with predicted_rate
    └── validation_predictions.csv  # FINAL OUTPUT: load_id,predicted_rate (12,000 rows)
```
```

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

1. Put the four input CSVs in `data/`.
2. Run the notebook top to bottom (random seeds are fixed, so results are reproducible):
   ```bash
   jupyter nbconvert --to notebook --execute Model.ipynb --inplace
   ```
   or open `Model.ipynb` in Jupyter and choose **Run All**.
3. Score the outputs:
   ```bash
   python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
   ```
   The scorer validates both files and writes `scorer_results/candidate_december.png`.

## Approach

**Validation split.** `validation.csv` is entirely in the future of the training data (train: Jan 1 - Oct 31 2025; validation: Nov 1 - Dec 31 2025), so a random split would overstate accuracy. Models are trained on Jan-Aug (38,477 rows) and tested on a Sep-Oct holdout (9,523 rows).

**Data quality.**

| Issue | Handling |
|---|---|
| 624 of 48,000 training rows have an implausible rate per mile (below $0.90 or above $5) | Excluded from training; metrics reported on all holdout rows and on clean rows |
| 292 negative weights (sign flips) | Absolute value, plus a flag |
| 300 missing weights | Median fill, plus a flag |
| 374 train / 249 validation missing `market_index` | Filled with that date's average (it barely varies within a day) |
| Reported distance inconsistent with city coordinates (187 train / 39 validation rows) | Replaced with 1.18 x straight-line distance and flagged. Distances at the 70-mile floor are left alone. |
| December chart rows have no coordinates, `market_index` or `quote_signal` | Coordinates from a city lookup; `market_index` from the date average in `validation.csv`; `quote_signal` from the monthly average |

**Features.** Log distance, straight-line distance, equipment, weight, market index, quote signal, pickup and delivery city, coordinates and coordinate differences, month, day of week, day of year.

**Models compared** (target is log of rate, predictions are converted back; holdout MAE in USD):

| Model | MAE (all rows) | MAE (clean rows) | RMSE (clean rows) | MAPE (clean rows) |
|---|---|---|---|---|
| Ridge (log) | 105.29 | 53.21 | 159.49 | 2.31% |
| Random Forest | 112.24 | 60.15 | 164.33 | 2.50% |
| LightGBM | 109.04 | 57.05 | 163.09 | 2.38% |
| **Hybrid: Ridge + LightGBM on residuals (final)** | **104.66** | **52.60** | **159.38** | **2.25%** |

Rate is largely linear in log distance with city and equipment effects, so Ridge beats the tree models alone. The hybrid adds a small gain by fitting a regularized LightGBM to Ridge's residuals. The final model is refit on all clean development rows before predicting.

## Limitations

- The December chart's `quote_signal` is a monthly average, so day-to-day movement in the chart comes mainly from `market_index`.
- Corrupted-label rows are only about 1.3% of training rows but roughly double holdout MAE (about 105 on all rows vs 53 on clean rows). No model can predict them, so the clean-row metrics show typical accuracy.
