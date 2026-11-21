import argparse
import json
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import ACTS, DAY, load_tables, build_events, cutoff_features, cutoff_labels

SETS = {
    "lgbm_base": [r"^base_"],
    "lgbm_multiscale": [r"^base_", r"^tmp_\w+_(1d|3d|14d)$", r"^tmp_total", r"^tmp_active"],
    "lgbm_recency": [r"^base_", r"_decay\d+$", r"_gap$", r"^tmp_gap_any", r"_trend$"],
    "lgbm_periodic": [r"^base_", r"^tmp_dow", r"^tmp_blk"],
    "lgbm_temporal": [r"^base_", r"^tmp_"],
    "lgbm_temporal_repo": [r"^base_", r"^tmp_", r"^rel_n_repos", r"^rel_repo_"],
    "lgbm_temporal_pr": [r"^base_", r"^tmp_", r"^rel_pr_"],
    "lgbm_temporal_issue": [r"^base_", r"^tmp_", r"^rel_issue_"],
    "lgbm_full": [r"^base_", r"^tmp_", r"^rel_"],
}
PARAMS = dict(n_estimators=200, learning_rate=0.07, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=4)


def f1_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def score(pred, truth):
    tp = ((pred == 1) & (truth == 1)).sum(axis=0)
    fp = ((pred == 1) & (truth == 0)).sum(axis=0)
    fn = ((pred == 0) & (truth == 1)).sum(axis=0)
    per = [f1_counts(a, b, c) for a, b, c in zip(tp, fp, fn)]
    macro = float(np.mean(per))
    micro = f1_counts(tp.sum(), fp.sum(), fn.sum())
    return {"per_event": dict(zip(ACTS, per)), "macro_f1": macro, "micro_f1": micro, "score": (macro + micro) / 2}


def best_thresholds(prob, truth):
    grid = np.arange(0.03, 0.95, 0.01)
    out = []
    for a in range(prob.shape[1]):
        scores = []
        for th in grid:
            pred = prob[:, a] >= th
            tp = (pred & (truth[:, a] == 1)).sum()
            fp = (pred & (truth[:, a] == 0)).sum()
            fn = ((~pred) & (truth[:, a] == 1)).sum()
            scores.append(f1_counts(tp, fp, fn))
        out.append(float(grid[int(np.argmax(scores))]))
    return np.array(out)


def select_columns(columns, patterns):
    return [c for c in columns if any(re.search(p, c) for p in patterns)]


def fit_predict(train_x, train_y, test_x, importances=None):
    prob = np.zeros((len(test_x), len(ACTS)))
    for a in range(len(ACTS)):
        if train_y[:, a].sum() == 0:
            continue
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(train_x, train_y[:, a])
        prob[:, a] = model.predict_proba(test_x)[:, 1]
        if importances is not None:
            gain = model.booster_.feature_importance(importance_type="gain")
            importances[ACTS[a]] = dict(zip(train_x.columns, (gain / max(gain.sum(), 1e-12)).tolist()))
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--test-input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-start", default="2026-11-21 16:00")
    ap.add_argument("--data-end")
    ap.add_argument("--min-history-days", type=int, default=28)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--mode", choices=["validate", "final"], default="validate")
    ap.add_argument("--student-id", default="315551802")
    args = ap.parse_args()

    tables = load_tables(args.db)
    events = build_events(tables)
    users_tab = tables["Users"]
    cohort = users_tab[users_tab["is_cohort"].astype(bool)]
    users = cohort["user_id"].to_numpy()
    first_seen = users_tab.set_index("user_id")["first_seen_at"]
    test_ids = pd.read_csv(args.test_input)["UserID"]
    code_to_uid = users_tab.set_index("github_user_id")["user_id"]
    mapped = code_to_uid.reindex(test_ids)
    known = mapped.notna().to_numpy()
    print(f"{known.sum()} of {len(test_ids)} test users appear in the tables, the others get all zeros")
    test_uids = mapped[known].to_numpy().astype(np.int64)
    test_mask = np.isin(users, test_uids)

    data_start = users_tab["first_seen_at"].min().floor("D")
    data_end = pd.Timestamp(args.data_end) if args.data_end else events["v"].max().floor("h")
    test_start = pd.Timestamp(args.test_start)
    gap = max(test_start - data_end, pd.Timedelta(0))
    print(f"data {data_start} -> {data_end}, gap to test week {gap}")

    last_label_cutoff = (data_end - gap - 7 * DAY).floor("h")
    first_cutoff = data_start + args.min_history_days * DAY
    cutoffs = []
    c = last_label_cutoff
    while c >= first_cutoff:
        cutoffs.append(c)
        c = c - DAY
    cutoffs = sorted(cutoffs)
    print(f"{len(cutoffs)} cutoffs from {cutoffs[0]} to {cutoffs[-1]}")

    frames, labels = [], []
    for c in cutoffs:
        x = cutoff_features(events, tables, users, c, first_seen)
        x["cutoff"] = c
        frames.append(x)
        labels.append(cutoff_labels(events, users, c + gap, c + gap + 7 * DAY))
    X = pd.concat(frames, ignore_index=True)
    cutoff_col = X["cutoff"].to_numpy()
    Y = np.vstack(labels)
    feature_cols = [c for c in X.columns if c != "cutoff"]
    print(f"{X.shape[0]} samples, {len(feature_cols)} features, positive rates {dict(zip(ACTS, Y.mean(axis=0).round(3)))}")

    fold_cutoffs = sorted(cutoffs[::-1][::7][:args.folds])
    results = {}
    oof = {name: [] for name in SETS}
    oof_truth, oof_copy = [], []
    tm_rows = np.tile(test_mask, 1)
    for fc in fold_cutoffs:
        train_sel = cutoff_col <= (fc - 7 * DAY).to_datetime64()
        val_sel = cutoff_col == fc.to_datetime64()
        if train_sel.sum() == 0:
            continue
        val_truth = Y[val_sel][tm_rows]
        copy_pred = (X.loc[val_sel, [f"base_{a}_7d" for a in ACTS]].to_numpy()[tm_rows] > 0).astype(int)
        oof_truth.append(val_truth)
        oof_copy.append(copy_pred)
        for name, prefixes in SETS.items():
            cols = select_columns(feature_cols, prefixes)
            prob = fit_predict(X.loc[train_sel, cols], Y[train_sel], X.loc[val_sel, cols])
            oof[name].append(prob[tm_rows])
        print(f"fold {fc} done", flush=True)

    truth = np.vstack(oof_truth)
    summary = {"copy_last_week": score(np.vstack(oof_copy), truth)}
    thresholds = {}
    for name in SETS:
        prob = np.vstack(oof[name])
        thresholds[name] = best_thresholds(prob, truth)
        summary[name] = score((prob >= thresholds[name]).astype(int), truth)
    for name, s in summary.items():
        print(f"{name:16s} macro {100*s['macro_f1']:.2f}  micro {100*s['micro_f1']:.2f}  score {100*s['score']:.2f}  per-activity {[round(float(100*v),1) for v in s['per_event'].values()]}")
    with open(f"{args.out}/validation_summary.json", "w") as f:
        json.dump({"summary": summary, "thresholds": {k: v.tolist() for k, v in thresholds.items()}, "folds": [str(c) for c in fold_cutoffs], "data_end": str(data_end), "gap": str(gap)}, f, indent=2)

    if args.mode == "final":
        cols = select_columns(feature_cols, SETS["lgbm_full"])
        final_x = cutoff_features(events, tables, users, data_end, first_seen)[cols]
        importances = {}
        prob = fit_predict(X[cols], Y, final_x, importances)
        pd.DataFrame(importances).fillna(0.0).assign(mean=lambda d: d.mean(axis=1)).sort_values("mean", ascending=False).to_csv(f"{args.out}/feature_importance.csv")
        full_oof = np.vstack(oof["lgbm_full"])
        target_rate = (full_oof >= thresholds["lgbm_full"]).mean(axis=0)
        test_prob = prob[test_mask]
        shifted = np.array([np.quantile(test_prob[:, a], 1.0 - target_rate[a]) for a in range(len(ACTS))])
        print("validation positive rate", dict(zip(ACTS, target_rate.round(3))))
        print("raw predicted rate on test users", dict(zip(ACTS, (test_prob >= thresholds["lgbm_full"]).mean(axis=0).round(3))))
        pred = (prob >= shifted).astype(int)
        frame = pd.DataFrame(pred, index=users, columns=ACTS)
        out = frame.reindex(mapped.fillna(-1).astype(np.int64).to_numpy()).fillna(0).astype(int)
        out.insert(0, "UserID", test_ids.to_numpy())
        out.to_csv(f"{args.out}/output_{args.student_id}.csv", index=False)
        print("wrote", f"{args.out}/output_{args.student_id}.csv", out.shape, dict(zip(ACTS, pred[test_mask].mean(axis=0).round(3))))


if __name__ == "__main__":
    main()
