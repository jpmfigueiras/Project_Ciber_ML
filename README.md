# Project_Ciber_ML

Machine-learning pipeline that detects anomalous TLS sessions in a post-quantum cryptography (PQC) network dataset. It covers the full workflow — exploratory analysis, data cleaning, feature engineering, clustering, class balancing, and training and comparing six classification models — and ends with a script that uses the trained models to classify new sessions.

Built as individual practical work for the Data Analysis and Cyber Intelligence course of the MSc in Cybersecurity at the Polytechnic Institute of Viana do Castelo.

## The problem

The dataset describes TLS handshakes captured under different scenarios: compliant sessions using post-quantum key exchange (ML-KEM-768 / ML-KEM-1024), classical sessions (X25519), and degraded or abnormal ones such as high network jitter, failed or incomplete connections and unexpected algorithm suites. Each session has to be classified as **Normal** or **Anomaly**.

Features include total bytes exchanged, record lengths, payload entropy, flow duration and timing, the negotiated algorithm suite, the TLS version and the connection outcome.

## Pipeline

1. **Exploratory analysis** (`notebooks/01_Exploratory_Analysis.ipynb`) — distributions, correlations, class balance and missing values per scenario.
2. **Missing values** — mean, median and constant imputation compared by cross-validated F1; constant imputation scored best.
3. **Normalisation** — Yeo-Johnson power transform on the skewed continuous features, then standard scaling.
4. **Feature engineering and selection** — derived security-relevant features (handshake-to-content entropy ratio, throughput in bytes per millisecond, client/server size asymmetry, a flag for legacy SSL 3.0 / TLS 1.0) followed by `SelectKBest` (ANOVA F-test).
5. **Clustering** — unsupervised K-Means to check how well the natural structure of the data matches the labels.
6. **Class balancing** — anomalies are the minority class, so SMOTE oversampling is applied to the training set only.
7. **Models** — K-Nearest Neighbours, Linear Regression (thresholded), Logistic Regression, Decision Tree, Random Forest and SVM, evaluated on a held-out test set and with cross-validation.
8. **Simulation** (`src/simulation.py`) — loads the saved models and classifies new, unseen sessions as a production system would.

## Results

Binary classification (Normal vs Anomaly) on the test set:

| Model                  | Accuracy | F1-score | CV accuracy | Training time |
|------------------------|---------:|---------:|------------:|--------------:|
| Random Forest          | 0.965    | 0.965    | 0.977       | 3.1 s         |
| Decision Tree          | 0.957    | 0.958    | 0.969       | 0.2 s         |
| Linear Regression      | 0.969    | 0.969    | 0.267       | 0.1 s         |
| K-Nearest Neighbours   | 0.932    | 0.936    | 0.944       | 0.3 s         |
| SVM                    | 0.913    | 0.920    | 0.904       | 25.4 s        |
| Logistic Regression    | 0.912    | 0.919    | 0.900       | 0.1 s         |

**Random Forest is the most reliable model**: it combines a high test F1-score with the best cross-validated accuracy. Linear Regression has the highest single test score, but its cross-validation accuracy collapses to 0.27, which shows the result does not generalise; it is a regressor turned into a classifier with a threshold and is kept only as a baseline.

## Running it

Requires Python 3.10 or later.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook                # run 01_Exploratory_Analysis, then 02_Full_Pipeline
python src/simulation.py        # classify new sessions with the trained models
```

The training script can also be run without the notebook:

```bash
python src/models.py
```

Trained models and result tables are regenerated on each run, so they are not tracked in Git.

## Project structure

```
Project_Ciber_ML/
├── data/raw/
│   ├── ml_features_and_labels.csv   # features and labels per session
│   ├── scenario_manifest.csv        # scenario, taxonomy and label per capture
│   └── raw_sessions.jsonl           # raw session records
├── notebooks/
│   ├── 01_Exploratory_Analysis.ipynb
│   └── 02_Full_Pipeline.ipynb
├── src/
│   ├── models.py        # preprocessing, training, evaluation and model saving
│   └── simulation.py    # loads saved models and classifies new sessions
└── requirements.txt
```

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), Matplotlib, Seaborn, Jupyter.
