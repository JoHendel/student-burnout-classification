"""
================================================================================
Burnout-Risk-Klassifikation  ·  "AI Impact on Students"
================================================================================
Binaere Klassifikation:  Burnout_Risk_Level  (High / Low)

Pipeline-Schritte:
    1. Laden + Datenqualitaets-Check
    2. Preprocessing (OneHot + Scaling) in sklearn-Pipeline
    3. Modellvergleich  LogReg / RandomForest / XGBoost  (5-fach CV)
    4. Schwellwert-Optimierung auf Recall der High-Klasse
       (Threshold wird per CV auf Trainingsdaten bestimmt -> kein Testleck)
    5. Finale Bewertung, Plots, Modell + Threshold speichern

Nutzung:
    pip install -r requirements.txt
    python src/burnout_pipeline.py
--------------------------------------------------------------------------------
Autor : Johannes Hendel (J0Tech / RUSH)
================================================================================
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     cross_val_predict, StratifiedKFold)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, precision_recall_curve)
from xgboost import XGBClassifier
import joblib

# ------------------------------------------------------------------ Konfiguration
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "AI_Impact_on_Student_-_Classification.csv"
PLOT_DIR = ROOT / "plots"; PLOT_DIR.mkdir(exist_ok=True)

TARGET        = "Burnout_Risk_Level"
POSITIVE      = "High"          # positive Klasse = Risiko, das gefunden werden soll
DROP_COLS     = ["Student_ID"]  # reiner Identifier, kein Signal
TARGET_RECALL = 0.85            # Mindest-Recall fuer High-Klasse (Praevention)
RANDOM_STATE  = 42
CV = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)


# ------------------------------------------------------------------ 1. Laden + Check
def load_and_check(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV nicht gefunden: {path}\n"
            "Datei von Kaggle laden und nach ./data/ legen "
            "(kaggle.com/datasets/dspritom/ai-impact-on-students)."
        )
    df = pd.read_csv(path)
    print("=" * 60)
    print("DATENQUALITAET")
    print("=" * 60)
    print(f"Shape             : {df.shape}")
    print(f"Fehlende Werte    : {int(df.isna().sum().sum())}")
    print(f"Duplikate (Zeile) : {int(df.duplicated().sum())}")
    print(f"Duplikate (ID)    : {int(df['Student_ID'].duplicated().sum())}")
    print(f"Klassenbalance    :\n{df[TARGET].value_counts(normalize=True).round(3)}\n")
    return df


# ------------------------------------------------------------------ 2. Preprocessing
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num = X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


# ------------------------------------------------------------------ 3. Modelle
def get_models() -> dict:
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight="balanced"),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1),
    }


def compare_models(X, y, pre) -> str:
    print("=" * 60)
    print("MODELLVERGLEICH  (Test-Split + 5-fach CV)")
    print("=" * 60)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"{'Modell':<14}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}  CV-AUC(5f)")
    best_name, best_auc = None, -1
    for name, clf in get_models().items():
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(Xtr, ytr)
        p  = pipe.predict(Xte)
        pr = pipe.predict_proba(Xte)[:, 1]
        cvs = cross_val_score(pipe, X, y, cv=CV, scoring="roc_auc", n_jobs=-1)
        print(f"{name:<14}{accuracy_score(yte,p):>7.3f}"
              f"{precision_score(yte,p):>7.3f}{recall_score(yte,p):>7.3f}"
              f"{f1_score(yte,p):>7.3f}{roc_auc_score(yte,pr):>7.3f}"
              f"  {cvs.mean():.3f}+/-{cvs.std():.3f}")
        if cvs.mean() > best_auc:
            best_auc, best_name = cvs.mean(), name
    print(f"\nBestes Modell nach CV-AUC: {best_name}\n")
    return best_name


# ------------------------------------------------------------------ 4. Schwellwert
def optimize_threshold(pipe, Xtr, ytr, target_recall: float):
    """Threshold per Out-of-Fold-Wahrscheinlichkeiten auf Trainingsdaten.
    Verhindert, dass der Schwellwert am Testset ueberangepasst wird."""
    oof = cross_val_predict(pipe, Xtr, ytr, cv=CV,
                            method="predict_proba", n_jobs=-1)[:, 1]
    prec, rec, thr = precision_recall_curve(ytr, oof)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    t_f1 = float(thr[np.argmax(f1[:-1])])
    mask = rec[:-1] >= target_recall
    t_rec = float(thr[mask][-1]) if mask.any() else 0.5
    return {"f1_optimal": t_f1, "high_recall": t_rec}, (prec, rec, thr)


def report_at(pr, yte, t, label):
    p = (pr >= t).astype(int)
    print(f"{label:<24} thr={t:.3f}  "
          f"Prec={precision_score(yte,p):.3f}  "
          f"Rec={recall_score(yte,p):.3f}  "
          f"F1={f1_score(yte,p):.3f}")
    return confusion_matrix(yte, p)


# ------------------------------------------------------------------ 5. Plots
def plot_threshold(prec, rec, thr, chosen, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr, prec[:-1], label="Precision", color="#264653")
    ax.plot(thr, rec[:-1], label="Recall", color="#d1495b")
    for name, t in chosen.items():
        ax.axvline(t, ls="--", alpha=.6,
                   label=f"{name} ({t:.2f})",
                   color="#2a9d8f" if name == "f1_optimal" else "#e76f51")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall vs. Threshold (High-Klasse)")
    ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close()


# ------------------------------------------------------------------ Main
def main():
    df = load_and_check(CSV_PATH)
    y = (df[TARGET] == POSITIVE).astype(int)
    X = df.drop(columns=[TARGET] + DROP_COLS)
    pre = build_preprocessor(X)

    best_name = compare_models(X, y, pre)

    # Finales Modell + Threshold-Tuning
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    clf = get_models()[best_name]
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    print("=" * 60)
    print("SCHWELLWERT-OPTIMIERUNG (Threshold per CV auf Trainingsdaten)")
    print("=" * 60)
    thresholds, (prec, rec, thr) = optimize_threshold(pipe, Xtr, ytr, TARGET_RECALL)

    pipe.fit(Xtr, ytr)
    pr = pipe.predict_proba(Xte)[:, 1]
    report_at(pr, yte, 0.50, "Standard (0.50)")
    report_at(pr, yte, thresholds["f1_optimal"], "F1-optimal")
    cm = report_at(pr, yte, thresholds["high_recall"],
                   f"High-Recall (>={TARGET_RECALL})")
    print(f"\nConfusion Matrix @High-Recall (rows=Low/High):\n{cm}\n")

    # Feature Importance (falls verfuegbar)
    model = pipe.named_steps["clf"]
    feat = pipe.named_steps["pre"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:  # LogReg -> Betrag der Koeffizienten
        imp = np.abs(model.coef_[0])
    top = (pd.Series(imp, index=[f.split("__", 1)[1] for f in feat])
           .sort_values(ascending=False).head(10))
    print("Top-10 Feature Importance:")
    print(top.round(4).to_string(), "\n")

    plot_threshold(prec, rec, thr, thresholds, PLOT_DIR / "threshold_tradeoff.png")

    # Modell + gewaehlter Threshold persistieren
    joblib.dump(pipe, ROOT / "burnout_model.joblib")
    (ROOT / "threshold.json").write_text(
        json.dumps({"model": best_name,
                    "recommended_threshold": thresholds["high_recall"],
                    "all_thresholds": thresholds}, indent=2))
    print(f"Gespeichert: burnout_model.joblib | threshold.json | "
          f"plots/threshold_tradeoff.png")


if __name__ == "__main__":
    main()
