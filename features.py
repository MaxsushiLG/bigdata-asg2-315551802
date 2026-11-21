import numpy as np
import pandas as pd

ACTS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
WINDOWS = [1, 3, 7, 14, 28]
HALF_LIVES = [1, 3, 7, 14]
MAX_AGE = 56.0
DAY = pd.Timedelta(days=1)


def to_time(series):
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def load_tables(db_dir):
    names = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments", "PRReviews", "PRReviewComments", "Pushes"]
    tables = {n: pd.read_parquet(f"{db_dir}/{n}.parquet") for n in names}
    time_cols = {
        "Users": ["first_seen_at"],
        "PullRequests": ["created_at", "merged_at", "closed_at", "first_seen_at"],
        "Issues": ["created_at", "closed_at", "first_seen_at"],
        "IssueComments": ["created_at"],
        "PRReviews": ["submitted_at", "dismissed_at", "first_seen_at"],
        "PRReviewComments": ["created_at"],
        "Pushes": ["created_at"],
    }
    for name, cols in time_cols.items():
        for col in cols:
            tables[name][col] = to_time(tables[name][col])
    return tables


def build_events(tables):
    parts = []
    pr = tables["PullRequests"]
    pr = pr[pr["author_id"].notna() & pr["created_at"].notna()]
    parts.append(pd.DataFrame({"uid": pr["author_id"], "act": 0, "repo": pr["repo_id"], "t": pr["created_at"], "seen": pr["first_seen_at"]}))
    iss = tables["Issues"]
    iss = iss[(~iss["is_pull_request"].astype(bool)) & iss["author_id"].notna() & iss["created_at"].notna()]
    parts.append(pd.DataFrame({"uid": iss["author_id"], "act": 1, "repo": iss["repo_id"], "t": iss["created_at"], "seen": iss["first_seen_at"]}))
    ic = tables["IssueComments"]
    parts.append(pd.DataFrame({"uid": ic["author_id"], "act": 2, "repo": ic["repo_id"], "t": ic["created_at"], "seen": ic["created_at"]}))
    rv = tables["PRReviews"]
    rv = rv[rv["reviewer_id"].notna()]
    t = rv["submitted_at"].fillna(rv["first_seen_at"])
    parts.append(pd.DataFrame({"uid": rv["reviewer_id"], "act": 3, "repo": rv["repo_id"], "t": t, "seen": rv["first_seen_at"]}))
    rc = tables["PRReviewComments"]
    parts.append(pd.DataFrame({"uid": rc["author_id"], "act": 4, "repo": rc["repo_id"], "t": rc["created_at"], "seen": rc["created_at"]}))
    pu = tables["Pushes"]
    parts.append(pd.DataFrame({"uid": pu["user_id"], "act": 5, "repo": pu["repo_id"], "t": pu["created_at"], "seen": pu["created_at"]}))
    events = pd.concat(parts, ignore_index=True)
    events = events[events["uid"].notna() & events["t"].notna()].copy()
    events["uid"] = events["uid"].astype(np.int64)
    events["v"] = events[["t", "seen"]].max(axis=1)
    return events.sort_values("t").reset_index(drop=True)


def count_matrix(frame, users):
    if len(frame) == 0:
        return pd.DataFrame(0.0, index=users, columns=range(6))
    grid = frame.groupby(["uid", "act"]).size().unstack(fill_value=0)
    return grid.reindex(index=users, columns=range(6), fill_value=0).astype(float)


def sum_matrix(frame, column, users):
    if len(frame) == 0:
        return pd.DataFrame(0.0, index=users, columns=range(6))
    grid = frame.groupby(["uid", "act"])[column].sum().unstack(fill_value=0.0)
    return grid.reindex(index=users, columns=range(6), fill_value=0.0)


def base_features(window, users):
    out = {}
    for w in (7, 28):
        m = count_matrix(window[window["age"] <= w], users)
        for a, name in enumerate(ACTS):
            out[f"base_{name}_{w}d"] = m[a]
    return pd.DataFrame(out, index=users)


def temporal_features(window, users, first_seen, cutoff):
    out = {}
    for w in (1, 3, 14):
        m = count_matrix(window[window["age"] <= w], users)
        for a, name in enumerate(ACTS):
            out[f"tmp_{name}_{w}d"] = m[a]
    m28 = count_matrix(window[window["age"] <= 28], users)
    m7 = count_matrix(window[window["age"] <= 7], users)
    for a, name in enumerate(ACTS):
        out[f"tmp_{name}_trend"] = (m7[a] + 1.0) / (m28[a] / 4.0 + 1.0)
    for hl in HALF_LIVES:
        frame = window[["uid", "act"]].copy()
        frame["d"] = np.exp(-window["age"].to_numpy() * np.log(2.0) / hl)
        m = sum_matrix(frame, "d", users)
        for a, name in enumerate(ACTS):
            out[f"tmp_{name}_decay{hl}"] = m[a]
    if len(window):
        gap = window.groupby(["uid", "act"])["age"].min().unstack().reindex(index=users, columns=range(6))
    else:
        gap = pd.DataFrame(np.nan, index=users, columns=range(6))
    for a, name in enumerate(ACTS):
        out[f"tmp_{name}_gap"] = gap[a].fillna(MAX_AGE)
    last_any = window.groupby("uid")["age"].min().reindex(users) if len(window) else pd.Series(np.nan, index=users)
    out["tmp_gap_any"] = last_any.fillna(MAX_AGE)
    w28 = window[window["age"] <= 28]
    total = w28.groupby("uid").size().reindex(users).fillna(0.0)
    out["tmp_total_28d"] = total
    shifted = w28["t"] + pd.Timedelta(hours=8)
    dow = pd.crosstab(w28["uid"], shifted.dt.dayofweek).reindex(index=users, columns=range(7), fill_value=0)
    blk = pd.crosstab(w28["uid"], shifted.dt.hour // 6).reindex(index=users, columns=range(4), fill_value=0)
    denom = total.replace(0, np.nan)
    for d in range(7):
        out[f"tmp_dow{d}"] = (dow[d] / denom).fillna(0.0)
    for b in range(4):
        out[f"tmp_blk{b}"] = (blk[b] / denom).fillna(0.0)
    active_days = w28.assign(day=w28["t"].dt.floor("D")).groupby("uid")["day"].nunique().reindex(users).fillna(0.0)
    out["tmp_active_days_28d"] = active_days
    out["tmp_days_since_first_seen"] = (cutoff - first_seen.reindex(users)).dt.total_seconds() / 86400.0
    return pd.DataFrame(out, index=users)


def relational_features(window, tables, users, cutoff):
    out = {}
    w28 = window[window["age"] <= 28]
    w7 = window[window["age"] <= 7]
    pairs = w28[["uid", "repo"]].dropna().drop_duplicates()
    repo7 = w7.groupby("repo").agg(rep_events=("uid", "size"), rep_users=("uid", "nunique"))
    pairs = pairs.join(repo7, on="repo")
    pairs[["rep_events", "rep_users"]] = pairs[["rep_events", "rep_users"]].fillna(0.0)
    grouped = pairs.groupby("uid")
    out["rel_n_repos_28d"] = grouped.size().reindex(users).fillna(0.0)
    out["rel_repo_events_sum"] = grouped["rep_events"].sum().reindex(users).fillna(0.0)
    out["rel_repo_events_max"] = grouped["rep_events"].max().reindex(users).fillna(0.0)
    out["rel_repo_users_sum"] = grouped["rep_users"].sum().reindex(users).fillna(0.0)
    out["rel_repo_users_max"] = grouped["rep_users"].max().reindex(users).fillna(0.0)
    pull = tables["PullRequests"]
    pull = pull[pull["author_id"].notna() & pull["created_at"].notna() & (pull["first_seen_at"] < cutoff) & (pull["created_at"] < cutoff) & (pull["created_at"] >= cutoff - 28 * DAY)].copy()
    rv = tables["PRReviews"]
    rv = rv[rv["first_seen_at"] < cutoff]
    rc = tables["PRReviewComments"]
    rc = rc[rc["created_at"] < cutoff]
    iss = tables["Issues"]
    ic = tables["IssueComments"]
    ic = ic[ic["created_at"] < cutoff]
    pr_issue = iss[iss["pull_request_id"].notna()][["issue_id", "pull_request_id"]]
    pr_comments = ic.merge(pr_issue, on="issue_id").groupby("pull_request_id").size()
    pull["n_rev"] = pull["pr_id"].map(rv.groupby("pr_id").size()).fillna(0.0)
    pull["n_rc"] = pull["pr_id"].map(rc.groupby("pr_id").size()).fillna(0.0)
    pull["n_ic"] = pull["pr_id"].map(pr_comments).fillna(0.0)
    pull["merged"] = (pull["merged_at"].notna() & (pull["merged_at"] < cutoff)).astype(float)
    pull["closed"] = (pull["closed_at"].notna() & (pull["closed_at"] < cutoff)).astype(float)
    pull["open"] = 1.0 - np.maximum(pull["merged"], pull["closed"])
    pgrp = pull.groupby("author_id")
    out["rel_pr_authored_28d"] = pgrp.size().reindex(users).fillna(0.0)
    for col in ["n_rev", "n_rc", "n_ic", "merged", "closed", "open"]:
        out[f"rel_pr_{col}"] = pgrp[col].sum().reindex(users).fillna(0.0)
    mine = iss[(~iss["is_pull_request"].astype(bool)) & iss["author_id"].notna() & iss["created_at"].notna() & (iss["first_seen_at"] < cutoff) & (iss["created_at"] < cutoff) & (iss["created_at"] >= cutoff - 28 * DAY)][["issue_id", "author_id", "closed_at"]].copy()
    others = ic.merge(mine[["issue_id", "author_id"]], on="issue_id", suffixes=("", "_issue"))
    others = others[others["author_id"] != others["author_id_issue"]]
    mine["n_in"] = mine["issue_id"].map(others.groupby("issue_id").size()).fillna(0.0)
    mine["open"] = (mine["closed_at"].isna() | (mine["closed_at"] >= cutoff)).astype(float)
    igrp = mine.groupby("author_id")
    out["rel_issue_authored_28d"] = igrp.size().reindex(users).fillna(0.0)
    out["rel_issue_comments_in"] = igrp["n_in"].sum().reindex(users).fillna(0.0)
    out["rel_issue_open"] = igrp["open"].sum().reindex(users).fillna(0.0)
    return pd.DataFrame(out, index=users)


def cutoff_features(events, tables, users, cutoff, first_seen):
    lo = cutoff - MAX_AGE * DAY
    window = events[(events["v"] < cutoff) & (events["t"] >= lo) & (events["t"] < cutoff)].copy()
    window["age"] = (cutoff - window["t"]).dt.total_seconds() / 86400.0
    parts = [
        base_features(window, users),
        temporal_features(window, users, first_seen, cutoff),
        relational_features(window, tables, users, cutoff),
    ]
    return pd.concat(parts, axis=1)


def cutoff_labels(events, users, start, end):
    sel = events[(events["t"] >= start) & (events["t"] < end)]
    grid = sel.groupby(["uid", "act"]).size().unstack(fill_value=0).reindex(index=users, columns=range(6), fill_value=0)
    return (grid > 0).astype(int).to_numpy()
