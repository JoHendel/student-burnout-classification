# Student Burnout Risk Classification

Binäre Klassifikation des **Burnout-Risikos** von Studierenden auf Basis ihrer
GenAI-Nutzung und weiterer Lern-Merkmale. End-to-End Machine-Learning-Pipeline
mit Datenqualitäts-Check, Modellvergleich, **Schwellwert-Optimierung** und
reproduzierbarer Auswertung.

> Dataset: [AI Impact on Students](https://www.kaggle.com/datasets/dspritom/ai-impact-on-students) (Kaggle)
> · 28.856 Zeilen · 15 Spalten · Ziel: `Burnout_Risk_Level` (High / Low)

---

## Ergebnisse

| Modell | Accuracy | F1 | ROC-AUC | CV-AUC (5-fach) |
|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0,790 | 0,748 | 0,864 | **0,865 ± 0,003** |
| Random Forest | 0,783 | 0,721 | 0,853 | 0,859 ± 0,004 |
| XGBoost | 0,789 | 0,732 | 0,861 | 0,863 ± 0,004 |

**Beste Wahl: Logistic Regression** — statistisch gleichauf mit XGBoost, aber
interpretierbar, schnell und robust (Occam's Razor). Sehr niedrige CV-Streuung
→ kein Overfitting.

### Schwellwert-Optimierung (High-Klasse)

Der Standard-Schwellwert 0,5 übersieht ~28 % der gefährdeten Studierenden.
Für einen Präventions-Anwendungsfall zählt **Recall**, nicht Accuracy:

| Threshold | Precision | Recall (High) | F1 | Verpasste Fälle |
|---|---|---|---|---|
| Standard 0,50 | 0,780 | 0,718 | 0,748 | 705 |
| F1-optimal 0,41 | 0,729 | 0,786 | **0,756** | 535 |
| **High-Recall 0,34** | 0,680 | **0,839** | 0,751 | **402** |

Der Schwellwert wird **per Cross-Validation auf den Trainingsdaten** bestimmt,
nicht am Testset — sonst wären die Werte optimistisch verzerrt.

---

## Wichtigste Einflussfaktoren

1. **Weekly_GenAI_Hours** — mit Abstand stärkster Prädiktor (r = +0,56)
2. **Perceived_AI_Dependency** — gefühlte KI-Abhängigkeit (r = +0,46)
3. `Year_of_Study` (Graduate / Freshman)
4. `Institutional_Policy = Strict_Ban`

Hohe GenAI-Nutzung korreliert mit **weniger** klassischem Lernen — GenAI scheint
es eher zu ersetzen als zu ergänzen.

---

## Projektstruktur

```
student-burnout-classification/
├── src/
│   └── burnout_pipeline.py      # komplette Pipeline (Laden → Modell → Threshold)
├── data/                        # CSV hier ablegen (nicht im Repo, s. .gitignore)
├── plots/                       # generierte Grafiken
├── requirements.txt
├── LICENSE
└── README.md
```

## Nutzung

```bash
# 1. Abhängigkeiten
pip install -r requirements.txt

# 2. Datensatz von Kaggle laden und nach data/ legen
#    kaggle.com/datasets/dspritom/ai-impact-on-students

# 3. Pipeline starten
python src/burnout_pipeline.py
```

Erzeugt: Modellvergleich + Report in der Konsole, `plots/threshold_tradeoff.png`,
sowie das trainierte Modell (`burnout_model.joblib`) und den empfohlenen
Schwellwert (`threshold.json`).

---

## Methodik

- **Preprocessing** in einer `sklearn.Pipeline` (OneHotEncoder für kategoriale,
  StandardScaler für numerische Features) → kein Data-Leakage zwischen Folds
- **Stratifizierter** 80/20-Split + 5-fach StratifiedKFold-CV
- **Datenleck geprüft**: `Post_Semester_GPA` entfernen ändert CV-AUC nicht (0,863)
- **Klassenbalance**: 57 / 43 → `class_weight='balanced'` statt Oversampling

## Grenzen & kritische Einordnung

- Der Datensatz ist auffällig sauber (0 fehlende Werte, 0 Duplikate) und
  **sehr wahrscheinlich synthetisch** → gut für ML-Übung, aber keine echten
  kausalen Schlüsse über reale Studierende.
- Korrelation ≠ Kausalität: hohe GenAI-Nutzung ist evtl. Symptom statt Ursache.
- **Bias-Risiko**: das Modell gewichtet Studienjahr (Graduate/Freshman) stark —
  bei realem Einsatz Gefahr der Stigmatisierung ganzer Gruppen.

## Nächste Schritte

- [ ] SHAP-Werte für ehrliche, richtungsbehaftete Feature-Erklärung
- [ ] Feature Engineering: `GenAI_zu_Lernzeit_Ratio`, `GPA_Delta`
- [ ] `Tool_Diversity` droppen (r ≈ 0)
- [ ] Hyperparameter-Tuning (GridSearchCV) für XGBoost

## Lizenz

[MIT](LICENSE) © 2026 Johannes Hendel
