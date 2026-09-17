# Telco Customer Churn — pipeline MLOps de bout en bout

Projet final du cours MLOps (M2 Campus Cyber). Objectif : un projet ML complet, reproductible et **100 % local**, où **MLflow trace tout** (paramètres, métriques, artefacts, dataset, modèle) et sert de **Model Registry**, avec un modèle **entraîné puis déployé** derrière une API.

| Étape | Outil | Fichier |
|---|---|---|
| Données (téléchargement, nettoyage, validation de schéma) | pandas | [src/data.py](src/data.py) |
| Preprocessing + modèle dans un seul objet | sklearn `Pipeline` + `ColumnTransformer` | [src/pipeline.py](src/pipeline.py) |
| Tuning + tracking + registry | `GridSearchCV`, `mlflow.autolog`, Model Registry (alias `@champion`) | [src/train.py](src/train.py) |
| Évaluation (ROC/PR/matrice de confusion, rapport) loggée sur le run | MLflow artifacts | [src/evaluate.py](src/evaluate.py) |
| Inférence batch | registry ou export local | [src/predict.py](src/predict.py) |
| Déploiement | FastAPI + Docker + docker compose (serveur MLflow + API) | [app/main.py](app/main.py), [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml) |
| Config centralisée | YAML | [configs/config.yaml](configs/config.yaml) |
| Tests | pytest (22 tests : data, pipeline, train/eval sur MLflow temporaire, API) | [tests/](tests/) |
| Orchestration / CI | Makefile, GitHub Actions (lint + tests + pipeline + build Docker) | [Makefile](Makefile), [.github/workflows/ci.yml](.github/workflows/ci.yml) |

**Dataset** : [Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) (IBM) — 7 043 clients, 19 features mixtes (numériques + catégorielles), cible binaire `Churn` (26 % de positifs). Téléchargé automatiquement par `make data`.

---

## Démarrage rapide

Prérequis : Python 3.12, `make` (Git Bash sous Windows), Docker (optionnel).

```bash
git clone https://github.com/fatimaamrch/PROJET_MLOps.git && cd PROJET_MLOps
make install        # dépendances (versions épinglées) + package en mode editable
make pipeline       # data -> train -> evaluate, tout est loggé dans MLflow
make mlflow-ui      # http://localhost:5000 : expériences, runs, registry
make serve          # http://localhost:8000/docs : API de prédiction
```

Toutes les commandes : `make help`.

```
  data           Download the raw CSV and build the clean dataset
  train          GridSearchCV + MLflow tracking + Model Registry (MODEL=logreg|random_forest)
  evaluate       Score the @champion model, log plots & report to MLflow
  predict        Batch inference. Usage: make predict INPUT=file.csv [OUTPUT=out.csv]
  pipeline       Run the whole pipeline end to end
  test / lint / format
  mlflow-ui / serve
  docker-build / docker-run / docker-up / docker-down
```

---

## Structure du repo

```
PROJET_MLOps/
├─ configs/config.yaml        # chemins, cible, features, grilles d'hyperparamètres, CV, MLflow
├─ src/
│  ├─ data.py                 # download -> clean -> validate -> data/processed/
│  ├─ pipeline.py             # build_pipeline(numeric, categorical, model_type)
│  ├─ train.py                # GridSearchCV + autolog + log_model + registry + export joblib
│  ├─ evaluate.py             # métriques/plots hold-out loggés sur le run d'entraînement
│  ├─ predict.py              # scoring batch d'un CSV
│  └─ utils.py                # config, setup MLflow, split, plots
├─ app/main.py                # FastAPI : /health, /model-info, /predict, /predict/batch
├─ tests/                     # pytest (fixtures : dataset synthétique + MLflow SQLite temporaire)
├─ .github/workflows/ci.yml   # CI : lint, tests, pipeline complet, build Docker
├─ Makefile · Dockerfile · docker-compose.yml · pyproject.toml · requirements.txt · .env.example
├─ data/        (gitignoré)   # raw/ et processed/
├─ mlruns/ mlflow.db (gitignorés)  # backend MLflow local (SQLite + artefacts)
├─ artifacts/   (gitignoré)   # export joblib du champion + model_info.json (embarqué dans l'image Docker)
└─ reports/     (gitignoré)   # plots + rapports produits par evaluate.py
```

---

## Le pipeline en détail

### 1. Données — `make data`
- Télécharge le CSV une seule fois (`data/raw/`), puis nettoie : `TotalCharges` texte → numérique (blancs → NaN), `SeniorCitizen` → catégorielle, suppression de l'identifiant client et des doublons.
- **Validation de schéma** : le script échoue immédiatement si une colonne attendue par la config manque ou si le label positif n'existe pas.

### 2. Pipeline scikit-learn — `src/pipeline.py`
```
Pipeline
├─ pre: ColumnTransformer
│   ├─ num: SimpleImputer(median) -> StandardScaler
│   └─ cat: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown="ignore")
└─ model: LogisticRegression | RandomForestClassifier
```
Le preprocessing vit **dans** le modèle : impossible d'avoir un écart entre entraînement et inférence (training/serving skew).

### 3. Entraînement + traçabilité MLflow — `make train`
- Split stratifié 80/20 (seed fixée), `GridSearchCV` en `StratifiedKFold(5)` sur la grille de la config, optimisant le ROC-AUC.
- Ce que MLflow enregistre pour chaque run :
  - **params** : tous les hyperparamètres du pipeline + grille + meilleurs params (`best_model__C`…)
  - **metrics** : score CV de chaque candidat (runs enfants), `best_cv_roc_auc`, métriques hold-out (`test_roc_auc`, `test_f1`…)
  - **artifacts** : `config/config.yaml`, `cv_results.csv`, courbes d'entraînement (autolog), le **modèle** avec signature d'entrée/sortie, exemple d'entrée et dépendances épinglées
  - **dataset** : empreinte (digest) + schéma du jeu d'entraînement (`mlflow.log_input`)
  - **tags** : `model_type`, `dataset`, `stage`
- **Model Registry** : chaque entraînement crée une nouvelle version de `TelcoChurnClassifier`, taguée (`model_type`, `test_roc_auc`) et pointée par l'alias **`@champion`**. L'API et `predict.py` chargent `models:/TelcoChurnClassifier@champion` — on change de modèle en production en déplaçant l'alias, sans toucher au code.
- Export `artifacts/model.joblib` + `model_info.json` pour l'image Docker (pas besoin de MLflow au runtime).

```bash
make train                       # logreg (config)
make train MODEL=random_forest   # crée une v2, déplace @champion dessus
```

### 4. Évaluation — `make evaluate`
Recharge le champion depuis le registry, rejoue le même split (hold-out jamais vu), et logge **sur le run d'origine** : `eval_*` métriques, ROC, PR, matrice de confusion, `classification_report.json`, `test_predictions.csv` (analyse d'erreurs).

### 5. Déploiement — `make serve` / Docker
API FastAPI avec validation Pydantic stricte (valeurs catégorielles autorisées, bornes numériques) :

| Endpoint | Rôle |
|---|---|
| `GET /health` | liveness + source du modèle chargé |
| `GET /model-info` | version, run_id, métriques du modèle servi |
| `POST /predict` | un client → `{churn_probability, churn, threshold}` |
| `POST /predict/batch` | liste de clients |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "tenure": 2, "MonthlyCharges": 85.7, "TotalCharges": 171.4, "gender": "Female",
  "SeniorCitizen": "0", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
  "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
  "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"}'
# {"churn_probability":0.91,"churn":true,"threshold":0.5}
```

Résolution du modèle (`MODEL_SOURCE=auto`) : registry MLflow d'abord, sinon export local `artifacts/model.joblib`.

**Docker (API seule)** — l'image embarque le modèle exporté :
```bash
make train && make docker-build && make docker-run
```

**Docker compose (serveur MLflow + API)** — stack locale proche de la prod :
```bash
make docker-up                                            # MLflow UI :5000, API :8000
MLFLOW_TRACKING_URI=http://localhost:5000 make pipeline   # loggue sur le serveur
docker compose restart api                                # l'API recharge @champion depuis le serveur
```

---

## Résultats (hold-out 20 %, seed 42)

| Modèle | Meilleurs params (CV 5 folds) | CV ROC-AUC | Test ROC-AUC | Accuracy | F1 (churn) | Recall (churn) |
|---|---|---|---|---|---|---|
| LogisticRegression (v1) | C=10, liblinear, l2 | 0.846 | **0.840** | 0.803 | 0.586 | 0.527 |
| RandomForest (v2) | 300 arbres, depth 10, leaf 5, balanced | 0.848 | **0.842** | 0.769 | 0.625 | 0.726 |

Les deux modèles sont équivalents en ROC-AUC ; le RandomForest `class_weight="balanced"` récupère beaucoup plus de churners (recall 0.73 vs 0.53) au prix d'une précision plus faible — choix à faire selon le coût métier d'un churn manqué vs d'une action de rétention inutile. Le seuil de décision est un paramètre (`evaluation.threshold`, ou `THRESHOLD` pour l'API).

---

## Bonnes pratiques MLOps appliquées

- **Reproductibilité** : config YAML unique, seeds fixées, split stratifié déterministe, versions de dépendances épinglées (`requirements.txt`) — un modèle picklé avec scikit-learn 1.7.2 est servi avec scikit-learn 1.7.2.
- **Traçabilité complète** : chaque run MLflow relie config + données (digest) + code (params) + métriques + modèle + version de registry ; l'API expose `run_id` et version via `/model-info`.
- **Pas de training/serving skew** : preprocessing dans le `Pipeline`, mêmes coercitions de types (`coerce_feature_types`) à l'entraînement et à l'inférence, signature MLflow.
- **Registry & promotion** : alias `@champion` découplé du code ; rollback = déplacer l'alias dans l'UI MLflow.
- **Qualité** : 22 tests (données, pipeline, tracking/registry sur backend temporaire, API), lint `ruff`, CI GitHub Actions qui rejoue tout le pipeline et construit l'image.
- **Séparation code / données / artefacts** : `data/`, `mlruns/`, `artifacts/`, `reports/` gitignorés et régénérables.
- **Config 12-factor** : variables d'environnement (`.env.example`) > `config.yaml`.
- **Conteneurisation** : image auto-suffisante avec healthcheck ; compose pour un serveur MLflow local.

---

## Tests et qualité

```bash
make test     # pytest (≈ 40 s : entraîne une fois sur un dataset synthétique + MLflow SQLite temporaire)
make lint     # ruff check + format --check
make format   # auto-fix
```

## Notes Windows

Le `Makefile` s'exécute avec un shell POSIX : utiliser **Git Bash** (`make` installable via `winget install ezwinports.make`). Les scripts Python fonctionnent aussi directement : `python src/train.py --config configs/config.yaml`.
