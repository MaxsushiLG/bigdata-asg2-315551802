# Predicting GitHub User Activity (Assignment 2, Big Data Analytics)

Maxime Le Glanaër, student ID 315551802

Data Source: GH Archive https://www.gharchive.org/

## Overview
For each of the 4,465 test users, the code predicts which of 6 activities (open_pr, open_issue, issue_comment, pr_review, pr_review_comment, push) the user performs during 2026-11-22 00:00 to 2026-11-29 00:00 (Taiwan time).
Data period used: 2026-07-01 to 2026-10-15 (UTC). Only data before the prediction time is used.

## Files
- `features.py`: loads the 8 tables, builds the event table, computes base, temporal and relational features and labels at a cutoff.
- `pipeline.py`: builds training samples at daily cutoffs, runs rolling time-based validation and ablation, trains one LightGBM model per activity, selects thresholds, writes the output CSV.
- `plot_coverage.py`: plots daily event counts per activity (data coverage check).
- `requirements.txt`: Python dependencies.

## How to run
1. Setup
```
git clone https://github.com/MaxsushiLG/bigdata-asg2-315551802.git
cd bigdata-asg2-315551802
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git course
```
2. Download GH Archive hourly files (2026-07-01 to 2026-10-14)
```
mkdir -p raw && for h in $(seq 0 2519); do t=$(date -u -d "2026-07-01 00:00 +$h hour" +%Y-%m-%d-%-H); wget -nc -q -P raw https://data.gharchive.org/$t.json.gz; done
```
3. Build the 8 tables
```
python3 course/scripts/preprocess.py --raw raw --out db
```
4. Validation and ablation
```
python3 pipeline.py --db db --test-input course/submission/test_input.csv --out out --data-end "2026-10-15 00:00" --mode validate
```
5. Final prediction
```
python3 pipeline.py --db db --test-input course/submission/test_input.csv --out out --data-end "2026-10-15 00:00" --mode final --student-id 315551802
```
Outputs: `out/output_315551802.csv`, `out/validation_summary.json`, `out/feature_importance.csv`.
6. Format check
```
python3 course/scripts/evaluate.py --pred out/output_315551802.csv --input course/submission/test_input.csv
```
7. Optional coverage plot
```
python3 plot_coverage.py --db db --out out/coverage.png
```
