# Student Burnout Risk Classification

## Kurzbeschreibung

Dieses Projekt implementiert eine binäre Klassifikation des Burnout-Risikos
von Studierenden (`High` / `Low`) auf Basis ihrer GenAI-Nutzung und weiterer
lernbezogener Merkmale. Es umfasst eine vollständige, reproduzierbare
Machine-Learning-Pipeline mit Datenqualitätsprüfung, Modellvergleich,
Schwellwert-Optimierung und Auswertung.

## Projektziel

Ziel ist es, gefährdete Studierende möglichst zuverlässig zu identifizieren.
Der Anwendungsfall ist präventiv ausgerichtet: Es ist wichtiger, tatsächlich
gefährdete Personen zu erkennen (hoher Recall der Klasse `High`), als eine
maximale Gesamtgenauigkeit zu erreichen. Die Pipeline optimiert den
Klassifikationsschwellwert entsprechend diesem Ziel.

## Datensatz

- **Quelle:** [AI Impact on Students](https://www.kaggle.com/datasets/dspritom/ai-impact-on-students) (Kaggle)
- **Umfang:** 28.856 Zeilen, 15 Spalten
- **Zielvariable:** `Burnout_Risk_Level` (`High` / `Low`)
- **Klassenverteilung:** 57 % `Low`, 43 % `High` (leicht unausgeglichen)

Der Datensatz enthält keine fehlenden Werte und keine Duplikate. Die Spalte
`Student_ID` ist ein reiner Identifier und wird vor dem Training entfernt.

Wichtigste Merkmale: `Weekly_GenAI_Hours`, `Perceived_AI_Dependency`,
`Traditional_Study_Hours`, `Anxiety_Level_During_Exams`, `Pre_Semester_GPA`,
`Post_Semester_GPA` sowie die kategorialen Variablen `Major_Category`,
`Year_of_Study`, `Primary_Use_Case`, `Prompt_Engineering_Skill`,
`Paid_Subscription` und `Institutional_Policy`.

Der Datensatz wird aus Lizenz- und Größengründen nicht mit dem Repository
ausgeliefert (siehe `.gitignore`). Er muss vor der Ausführung von Kaggle
geladen und im Verzeichnis `data/` abgelegt werden.

## Vorgehensweise

Die Pipeline durchläuft folgende Schritte:

1. Laden der Daten und Prüfung der Datenqualität (fehlende Werte, Duplikate,
   Klassenverteilung).
2. Preprocessing innerhalb einer `scikit-learn`-Pipeline (OneHot-Encoding für
   kategoriale, Standardisierung für numerische Merkmale).
3. Modellvergleich mit 5-facher stratifizierter Kreuzvalidierung.
4. Schwellwert-Optimierung auf den Recall der Klasse `High`.
5. Finale Bewertung, Erzeugung der Plots sowie Speicherung von Modell und
   gewähltem Schwellwert.

## Explorative Datenanalyse

Die stärksten Zusammenhänge mit einem hohen Burnout-Risiko zeigen sich bei:

- `Weekly_GenAI_Hours` (Korrelation r = +0,56)
- `Perceived_AI_Dependency` (r = +0,46)

Eine höhere GenAI-Nutzung geht im Datensatz mit geringeren klassischen
Lernzeiten einher. Weitere relevante Merkmale sind das Studienjahr
(`Year_of_Study`) sowie eine strikte institutionelle Nutzungsregelung
(`Institutional_Policy = Strict_Ban`).

## Machine-Learning-Ansatz

Verglichen werden drei Modelle:

- **Logistic Regression** (Baseline, interpretierbar)
- **Random Forest** (nicht-linear, robust)
- **XGBoost** (Gradient Boosting)

Das Preprocessing erfolgt innerhalb der Pipeline, wodurch Data-Leakage zwischen
den Kreuzvalidierungs-Folds vermieden wird. Die Modellauswahl basiert auf der
mittleren ROC-AUC aus 5-facher stratifizierter Kreuzvalidierung.

Der Klassifikationsschwellwert wird über Out-of-Fold-Wahrscheinlichkeiten auf
den Trainingsdaten bestimmt und nicht am Testset. Dadurch werden optimistisch
verzerrte Ergebnisse vermieden.

## Ergebnisse

Modellvergleich (Test-Split und 5-fache Kreuzvalidierung):

| Modell               | Accuracy | F1    | ROC-AUC | CV-AUC (5-fach)   |
| -------------------- | -------- | ----- | ------- | ----------------- |
| Logistic Regression  | 0,790    | 0,748 | 0,864   | 0,865 ± 0,003     |
| Random Forest        | 0,783    | 0,721 | 0,853   | 0,859 ± 0,004     |
| XGBoost              | 0,789    | 0,732 | 0,861   | 0,863 ± 0,004     |

Ausgewähltes Modell: **Logistic Regression**. Es liefert eine mit XGBoost
vergleichbare Leistung, ist jedoch interpretierbar und rechnerisch günstiger.
Die geringe Streuung der Kreuzvalidierung deutet auf kein Overfitting hin.

Schwellwert-Optimierung für die Klasse `High`:

| Schwellwert          | Precision | Recall (High) | F1    | Verpasste Fälle |
| -------------------- | --------- | ------------- | ----- | --------------- |
| Standard 0,50        | 0,780     | 0,718         | 0,748 | 705             |
| F1-optimal 0,41      | 0,729     | 0,786         | 0,756 | 535             |
| High-Recall 0,34     | 0,680     | 0,839         | 0,751 | 402             |

Der Standard-Schwellwert übersieht rund 28 % der gefährdeten Studierenden. Mit
dem High-Recall-Schwellwert steigt der Anteil erkannter Fälle auf etwa 84 %,
bei entsprechend geringerer Precision.

## Projektstruktur

```
student-burnout-classification/
├── src/
│   └── burnout_pipeline.py      # Vollständige Pipeline (Laden → Modell → Threshold)
├── data/                        # CSV hier ablegen (nicht im Repository enthalten)
├── plots/                       # Generierte Grafiken
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Anschließend den Datensatz von Kaggle laden und die CSV-Datei im Verzeichnis
`data/` ablegen:
[https://www.kaggle.com/datasets/dspritom/ai-impact-on-students](https://www.kaggle.com/datasets/dspritom/ai-impact-on-students)

## Nutzung

```bash
python src/burnout_pipeline.py
```

Ausgabe:

- Modellvergleich und Auswertung in der Konsole
- Grafik `plots/threshold_tradeoff.png`
- Trainiertes Modell `burnout_model.joblib`
- Empfohlener Schwellwert `threshold.json`

## Reproduzierbarkeit

- Fester Zufalls-Seed (`RANDOM_STATE = 42`) für Split, Kreuzvalidierung und
  Modelle
- Stratifizierter 80/20-Split sowie 5-fache stratifizierte Kreuzvalidierung
- Vollständiges Preprocessing innerhalb der Pipeline
- Schwellwert-Bestimmung ausschließlich auf Trainingsdaten

Es wurde geprüft, dass das Entfernen von `Post_Semester_GPA` die CV-AUC nicht
verändert (0,863); ein Data-Leakage über dieses Merkmal ist damit nicht
festzustellen.

## Grenzen des Projekts

- Der Datensatz ist auffällig sauber (keine fehlenden Werte, keine Duplikate)
  und mit hoher Wahrscheinlichkeit synthetisch. Er eignet sich für Übungs- und
  Demonstrationszwecke, erlaubt jedoch keine kausalen Schlüsse über reale
  Studierende.
- Korrelation impliziert keine Kausalität. Eine hohe GenAI-Nutzung kann
  Symptom statt Ursache sein.
- Das Modell gewichtet das Studienjahr stark. In einem realen Einsatz besteht
  dadurch das Risiko einer Benachteiligung einzelner Gruppen.

## Weiterentwicklung

- SHAP-Analyse zur richtungsbehafteten Erklärung der Merkmale
- Feature Engineering, z. B. Verhältnis von GenAI- zu Lernzeit oder GPA-Differenz
- Entfernen von `Tool_Diversity` (Korrelation nahe null)
- Hyperparameter-Optimierung für XGBoost.

## Verwendete Technologien

- Python
- pandas, NumPy
- scikit-learn
- XGBoost
- matplotlib
- joblib

Die konkreten Mindestversionen sind in `requirements.txt` festgelegt.

## Hinweis zu unterstützenden Werkzeugen

Bei der Planung, Strukturierung und Dokumentation des Projekts wurden
unterstützende KI-Werkzeuge verwendet. Die fachliche Prüfung, Anpassung und
finale Umsetzung erfolgte eigenständig.

## Lizenz

[MIT](LICENSE)
