"""Reproduce the Titanic experiments, metrics, and figures."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import argparse
import hashlib
import json
import platform
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder,
    RobustScaler, StandardScaler,
)

SEED = 42
ROOT = Path(__file__).resolve().parent
URL = ("https://raw.githubusercontent.com/datasciencedojo/datasets/"
       "f0ccab6a7ceafdff780052166fb6fab3311398eb/titanic.csv")
RAW_NUMERIC = ["Pclass", "SibSp", "Parch", "Fare"]
NUMERIC = [*RAW_NUMERIC, "Age"]
CATEGORICAL = ["Sex", "Embarked"]
DERIVED = ["FamilySize", "FarePerPerson"]
DEFAULT_OUTPUT = ROOT / "results"
SCORING = {"auc": "roc_auc", "f1": "f1"}
MAIN = ["Base", "Exp-1", "Exp-2", "Exp-3"]


@dataclass(frozen=True)
class Experiment:
    imputation: str = "mean"
    encoding: str = "onehot"
    scaler: str = "standard"
    k: int | None = None
    derived: bool = True


REFERENCE = Experiment()
CONFIGS = {
    "Base": Experiment("none", "none", "none", None, False),
    "Exp-1": REFERENCE,
    "Exp-2": Experiment("median", "label", "minmax", 6),
    "Exp-3": Experiment("most_frequent", "onehot", "robust", 6),
    "Median": replace(REFERENCE, imputation="median"),
    "Mode": replace(REFERENCE, imputation="most_frequent"),
    "Label": replace(REFERENCE, encoding="label"),
    "MinMax": replace(REFERENCE, scaler="minmax"),
    "Robust": replace(REFERENCE, scaler="robust"),
    "No-Scaling": replace(REFERENCE, scaler="none"),
    "Exp-2-All": Experiment("median", "label", "minmax"),
    "Exp-3-All": Experiment("most_frequent", "onehot", "robust"),
    "No-Derived": replace(REFERENCE, derived=False),
}

PAIRS = [
    ("Exp-1", "Median"), ("Exp-1", "Mode"), ("Exp-1", "Label"),
    ("Exp-1", "MinMax"), ("Exp-1", "Robust"), ("Exp-1", "No-Scaling"),
    ("Exp-2-All", "Exp-2"), ("Exp-3-All", "Exp-3"), ("No-Derived", "Exp-1"),
]


def add_features(frame):
    family_size = frame["SibSp"] + frame["Parch"] + 1
    return frame.assign(FamilySize=family_size, FarePerPerson=frame["Fare"] / family_size)


def make_pipeline(config, model):
    if config.encoding == "none":
        preprocess = ColumnTransformer([("raw", "passthrough", RAW_NUMERIC)])
    else:
        scalers = {
            "standard": StandardScaler(), "minmax": MinMaxScaler(),
            "robust": RobustScaler(), "none": "passthrough",
        }
        encoders = {
            "onehot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            "label": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        }
        num = Pipeline([
            ("impute", SimpleImputer(strategy=config.imputation)),
            ("scale", scalers[config.scaler]),
        ])
        cat = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", encoders[config.encoding]),
        ])
        numeric = NUMERIC + (DERIVED if config.derived else [])
        preprocess = ColumnTransformer([("num", num, numeric), ("cat", cat, CATEGORICAL)])

    classifier = {
        "LR": LogisticRegression(max_iter=3000, random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
    }[model]
    return Pipeline([
        ("derive", FunctionTransformer(add_features, validate=False) if config.derived else "passthrough"),
        ("preprocess", preprocess),
        ("select", SelectKBest(f_classif, k=config.k) if config.k else "passthrough"),
        ("model", classifier),
    ])


def holdout_metrics(estimator, X, y):
    prediction = estimator.predict(X)
    probability = estimator.predict_proba(X)[:, 1]
    return dict(
        accuracy=accuracy_score(y, prediction),
        precision=precision_score(y, prediction, zero_division=0),
        recall=recall_score(y, prediction, zero_division=0),
        f1=f1_score(y, prediction, zero_division=0),
        auc=roc_auc_score(y, probability),
    )


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def eda(data, train, output):
    missing = train.isna().sum()
    missing_table = pd.DataFrame({
        "all_missing": data.isna().sum(), "train_missing": missing,
        "train_pct": train.isna().mean() * 100,
    })
    missing_table.to_csv(output / "missing.csv", index_label="column")
    observed = train[["Age", "Fare", "SibSp", "Parch"]]
    q1, q3 = observed.quantile(0.25), observed.quantile(0.75)
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    outliers = observed.lt(lower) | observed.gt(upper)
    pd.DataFrame({
        "observed": observed.count(), "q1": q1, "q3": q3, "lower": lower,
        "upper": upper, "outliers": outliers.sum(),
    }).to_csv(output / "outliers.csv", index_label="column")
    corr = train[NUMERIC + ["Survived"]].corr(method="pearson")
    corr.to_csv(output / "correlation.csv", index_label="column")

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.0), layout="constrained")
    for index, variable in enumerate(["Age", "Fare"]):
        sns.histplot(data=train, x=variable, bins=26, color="#147d82", ax=axes[0, index])
        axes[0, index].set_title(f"{variable} distribution", loc="left", weight="bold")
        sns.boxplot(data=train, x=variable, color="#66b8b3", ax=axes[1, index], fliersize=2)
        axes[1, index].set_title(f"{variable} boxplot", loc="left", weight="bold")
    save_figure(fig, output, "distributions")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3), layout="constrained")
    sns.countplot(data=train, x="Survived", color="#147d82", ax=axes[0])
    axes[0].bar_label(axes[0].containers[0], padding=3)
    axes[0].set_xticks([0, 1], ["0 · Died", "1 · Survived"])
    axes[0].set_ylim(0, 500)
    axes[0].set_title("Target class counts", loc="left", weight="bold")
    sns.barplot(data=train, x="Sex", y="Survived", errorbar=None, color="#147d82", ax=axes[1])
    axes[1].bar_label(axes[1].containers[0], fmt="%.2f", padding=3)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Survival rate")
    axes[1].set_title("Survival rate by sex", loc="left", weight="bold")
    save_figure(fig, output, "target")

    fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    sns.heatmap(corr, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                cmap="vlag", linewidths=0.5, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Numeric feature and target correlations", loc="left", weight="bold")
    save_figure(fig, output, "correlation")

    return {
        "shape": list(data.shape), "duplicates": int(data.duplicated().sum()),
        "all_classes": data["Survived"].value_counts().to_dict(),
        "train_classes": train["Survived"].value_counts().to_dict(),
        "sex_survival": train.groupby("Sex")["Survived"].mean().to_dict(),
        "class_survival": train.groupby("Pclass")["Survived"].mean().to_dict(),
        "train_fare_median": train["Fare"].median(), "train_fare_max": train["Fare"].max(),
    }


def plot_selection(rankings, output):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), layout="constrained")
    for ax, (name, ranks) in zip(axes, rankings.groupby("experiment", sort=False)):
        ranked = ranks.sort_values("score")
        ax.barh(ranked["feature"], ranked["score"],
                color=np.where(ranked["selected"], "#147d82", "#bac5d1"))
        ax.set_title(f"{name} · top 6 in teal", loc="left", weight="bold")
        ax.set_xlabel("ANOVA F statistic")
    save_figure(fig, output, "selection")


def plot_finalists(finalists, X_test, y_test, output):
    fig, ax = plt.subplots(figsize=(5.8, 3.7), layout="constrained")
    for model, name, estimator in finalists:
        probabilities = estimator.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc = roc_auc_score(y_test, probabilities)
        ax.plot(fpr, tpr, label=f"{model} / {name}  ({auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="#9aa7b5", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right", fontsize=9)
    save_figure(fig, output, "roc")

    _, name, rf = next(row for row in finalists if row[0] == "RF")
    raw_columns = RAW_NUMERIC if name == "Base" else NUMERIC + CATEGORICAL
    # Pass raw columns through the fitted pipeline so permutation updates dependent features.
    importance = permutation_importance(
        rf, X_test[raw_columns], y_test, scoring="roc_auc", n_repeats=25,
        random_state=SEED, n_jobs=1,
    )
    rank = pd.DataFrame({
        "feature": raw_columns, "mean": importance.importances_mean,
        "std": importance.importances_std,
    }).sort_values("mean", ascending=False)
    rank.to_csv(output / "permutation.csv", index=False)
    ranked = rank.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.8, 3), layout="constrained")
    ax.barh(ranked["feature"], ranked["mean"], xerr=ranked["std"],
            color="#147d82", alpha=0.9, error_kw={"elinewidth": 1, "capsize": 2})
    ax.axvline(0, color="#667789", linewidth=1)
    ax.set_xlabel("ROC-AUC decrease after permutation")
    save_figure(fig, output, "permutation")


def run(csv_path, output):
    output.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=30) as response:
            csv_path.write_bytes(response.read())
    data = pd.read_csv(csv_path)
    print(f"Data shape = {data.shape}")
    print(data.dtypes.to_string())
    assert len(data) >= 500 and data["Survived"].notna().all()
    X = data.drop(columns="Survived")
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X_train, y_train))
    sns.set_theme(style="whitegrid", palette=["#147d82", "#ce743f"])
    stats = eda(data, data.loc[X_train.index], output)
    stats.update(
        train_rows=len(X_train), test_rows=len(X_test),
        test_classes=y_test.value_counts().to_dict(),
        sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        python=platform.python_version(), sklearn=sklearn.__version__, seed=SEED,
    )
    (output / "data_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    cv_rows, holdout_rows, selection_rows = [], [], []
    for name, config in CONFIGS.items():
        for model in ("LR", "RF"):
            estimator = make_pipeline(config, model)
            scores = cross_validate(
                estimator, X_train, y_train, cv=folds, scoring=SCORING,
                return_train_score=True, error_score="raise", n_jobs=1,
            )
            for fold in range(len(folds)):
                cv_rows.append(dict(experiment=name, model=model, fold=fold + 1,
                                    train_auc=scores["train_auc"][fold],
                                    val_auc=scores["test_auc"][fold],
                                    val_f1=scores["test_f1"][fold]))
            if name in MAIN:
                estimator.fit(X_train, y_train)
                holdout_rows.append(dict(experiment=name, model=model,
                                         **holdout_metrics(estimator, X_test, y_test)))
                if model == "LR" and config.k:
                    selector = estimator.named_steps["select"]
                    names = estimator.named_steps["preprocess"].get_feature_names_out()
                    for feature, score, selected in zip(names, selector.scores_, selector.get_support()):
                        selection_rows.append(dict(experiment=name,
                                                   feature=feature.split("__", 1)[-1],
                                                   score=score, selected=selected))
        print(f"Finished {name}", flush=True)

    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(output / "cv_folds.csv", index=False)
    pairs = []
    for before, after in PAIRS:
        for model in ("LR", "RF"):
            a = cv_frame[(cv_frame["experiment"] == before) & (cv_frame["model"] == model)].set_index("fold")
            b = cv_frame[(cv_frame["experiment"] == after) & (cv_frame["model"] == model)].set_index("fold")
            diff = b["val_auc"] - a["val_auc"]
            pairs.append(dict(before=before, after=after, model=model,
                              delta_auc=diff.mean(), delta_f1=(b["val_f1"] - a["val_f1"]).mean(),
                              fold_wins=int((diff > 1e-12).sum()),
                              before_gap=(a["train_auc"] - a["val_auc"]).mean(),
                              after_gap=(b["train_auc"] - b["val_auc"]).mean()))
    pd.DataFrame(pairs).to_csv(output / "paired.csv", index=False)
    ranking = pd.DataFrame(selection_rows).sort_values(["experiment", "score"], ascending=[True, False])
    ranking.to_csv(output / "selection.csv", index=False)
    plot_selection(ranking, output)

    grids = {
        "LR": {"model__C": [0.1, 1.0, 10.0]},
        "RF": {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 3, 6]},
    }
    mean_auc = cv_frame.groupby(["model", "experiment"])["val_auc"].mean()
    finalists, tuning, grid_rows = [], [], []
    for model, grid in grids.items():
        preset = max(MAIN, key=lambda name: mean_auc.loc[(model, name)])
        search = GridSearchCV(
            make_pipeline(CONFIGS[preset], model), grid, scoring="roc_auc",
            cv=folds, refit=True, error_score="raise", n_jobs=1,
        )
        search.fit(X_train, y_train)
        finalists.append((model, preset, search.best_estimator_))
        tuning.append(dict(model=model, preset=preset, cv_auc=search.best_score_,
                           params=search.best_params_,
                           **holdout_metrics(search, X_test, y_test)))
        for params, score, rank in zip(search.cv_results_["params"],
                                       search.cv_results_["mean_test_score"],
                                       search.cv_results_["rank_test_score"]):
            grid_rows.append(dict(model=model, preset=preset, params=params, auc=score, rank=int(rank)))
    plot_finalists(finalists, X_test, y_test, output)

    pd.DataFrame(holdout_rows).to_csv(output / "holdout.csv", index=False)
    (output / "tuned.json").write_text(json.dumps(tuning, indent=2), encoding="utf-8")
    (output / "grid.json").write_text(json.dumps(grid_rows, indent=2), encoding="utf-8")
    (output / "experiments.json").write_text(
        json.dumps({name: asdict(config) for name, config in CONFIGS.items()}, indent=2), encoding="utf-8",
    )
    print(pd.DataFrame(holdout_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "titanic.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.csv, args.output)

