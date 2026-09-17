"""Build the project report (docs/rapport.html + docs/rapport.pdf).

Numbers come from the MLflow registry / runs and the figures from reports/
(EDA notebook + evaluate.py), so the report is always consistent with the
last pipeline run.

Usage:
    python scripts/build_report.py            # HTML + PDF (PDF needs Chrome or Edge)
    python scripts/build_report.py --no-pdf
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import mlflow

from src.utils import PROJECT_ROOT, load_config, setup_mlflow

DOCS = PROJECT_ROOT / "docs"
FIGURES = PROJECT_ROOT / "reports" / "figures"
REPORTS = PROJECT_ROOT / "reports"

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
    "msedge",
]


# ---------- data collection ----------


def img(path: Path, width: str = "100%") -> str:
    if not path.exists():
        return f'<p class="missing">figure manquante : {path.name}</p>'
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:{width}" alt="{path.stem}">'


def collect_models(cfg: dict) -> list[dict]:
    """One row per registered version, with its run metrics and params."""
    setup_mlflow(cfg)
    client = mlflow.MlflowClient()
    name = cfg["mlflow"]["registered_model_name"]
    alias = cfg["mlflow"]["alias"]
    try:
        champion = client.get_model_version_by_alias(name, alias).version
    except Exception:  # noqa: BLE001
        champion = None
    rows = []
    for mv in client.search_model_versions(f"name='{name}'"):
        run = client.get_run(mv.run_id)
        m, p = run.data.metrics, run.data.params
        best = {k.removeprefix("best_model__"): v for k, v in p.items() if k.startswith("best_model__")}
        rows.append(
            {
                "version": int(mv.version),
                "champion": str(mv.version) == str(champion),
                "model_type": mv.tags.get("model_type", p.get("model_type", "?")),
                "run_id": mv.run_id,
                "best_params": ", ".join(f"{k}={v}" for k, v in sorted(best.items())),
                "cv_auc": m.get("best_cv_roc_auc"),
                "auc": m.get("test_roc_auc"),
                "acc": m.get("test_accuracy"),
                "prec": m.get("test_precision"),
                "rec": m.get("test_recall"),
                "f1": m.get("test_f1"),
            }
        )
    return sorted(rows, key=lambda r: r["version"])


def collect_eval(cfg: dict) -> dict:
    path = REPORTS / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def collect_data_stats(cfg: dict) -> dict:
    import pandas as pd

    from src.utils import load_processed

    df = load_processed(cfg)
    raw = pd.read_csv(PROJECT_ROOT / cfg["data"]["raw_path"])
    return {
        "raw_rows": len(raw),
        "rows": len(df),
        "n_num": len(cfg["features"]["numeric"]),
        "n_cat": len(cfg["features"]["categorical"]),
        "churn_rate": (df[cfg["data"]["target"]] == cfg["data"]["positive_label"]).mean(),
    }


# ---------- rendering ----------


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def models_table(rows: list[dict]) -> str:
    body = ""
    for r in rows:
        star = " <span class='badge'>@champion</span>" if r["champion"] else ""
        body += (
            f"<tr><td>v{r['version']}{star}</td><td>{r['model_type']}</td>"
            f"<td class='small'>{r['best_params']}</td>"
            f"<td>{fmt(r['cv_auc'])}</td><td><b>{fmt(r['auc'])}</b></td><td>{fmt(r['acc'])}</td>"
            f"<td>{fmt(r['prec'])}</td><td>{fmt(r['rec'])}</td><td>{fmt(r['f1'])}</td></tr>"
        )
    return f"""<table>
<thead><tr><th>Version</th><th>Modèle</th><th>Meilleurs hyperparamètres (CV)</th>
<th>CV ROC-AUC</th><th>Test ROC-AUC</th><th>Accuracy</th><th>Précision</th><th>Rappel</th><th>F1</th></tr></thead>
<tbody>{body}</tbody></table>"""


def render(cfg: dict, models: list[dict], ev: dict, ds: dict) -> str:
    champion = next((m for m in models if m["champion"]), models[-1] if models else None)
    champ_txt = (
        f"v{champion['version']} ({champion['model_type']}, ROC-AUC test {fmt(champion['auc'])})"
        if champion
        else "aucun"
    )
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<title>Rapport — Pipeline MLOps Telco Churn</title>
<style>
  @page {{ size: A4; margin: 18mm 16mm; }}
  body {{ font-family: "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2933; line-height: 1.45; font-size: 11pt; max-width: 900px; margin: 0 auto; padding: 0 16px; }}
  h1 {{ font-size: 24pt; margin-bottom: 0; color: #0b3954; }}
  h2 {{ font-size: 16pt; color: #0b3954; border-bottom: 2px solid #0b3954; padding-top: 14px; margin-top: 26px; page-break-after: avoid; }}
  h3 {{ font-size: 12.5pt; color: #1f4e79; margin-bottom: 4px; page-break-after: avoid; }}
  .subtitle {{ color: #52606d; margin-top: 4px; }}
  .meta {{ font-size: 9.5pt; color: #52606d; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0 14px; font-size: 9.5pt; page-break-inside: avoid; }}
  th, td {{ border: 1px solid #cbd2d9; padding: 5px 7px; text-align: left; vertical-align: top; }}
  th {{ background: #e4eaf1; }}
  td.small {{ font-size: 8.5pt; }}
  .badge {{ background: #0b3954; color: white; border-radius: 4px; padding: 1px 6px; font-size: 8pt; }}
  .kpis {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }}
  .kpi {{ flex: 1 1 130px; border: 1px solid #cbd2d9; border-radius: 6px; padding: 8px 10px; background: #f7f9fb; }}
  .kpi .v {{ font-size: 18pt; font-weight: 700; color: #0b3954; }}
  .kpi .l {{ font-size: 8.5pt; color: #52606d; }}
  figure {{ margin: 10px 0 16px; page-break-inside: avoid; text-align: center; }}
  figcaption {{ font-size: 9pt; color: #52606d; margin-top: 4px; }}
  .row {{ display: flex; gap: 12px; }}
  .row figure {{ flex: 1; }}
  code, pre {{ font-family: Consolas, "Courier New", monospace; font-size: 9pt; }}
  pre {{ background: #f1f4f8; padding: 8px 10px; border-radius: 4px; overflow-x: hidden; white-space: pre-wrap; font-size: 8.5pt; page-break-inside: avoid; }}
  ul {{ margin-top: 4px; }}
  .missing {{ color: #b00020; font-style: italic; }}
  .pb {{ page-break-before: always; }}
</style></head><body>

<h1>Pipeline MLOps de bout en bout — Telco Customer Churn</h1>
<p class="subtitle">Projet final du cours MLOps (M2 Campus Cyber) — Fatima Amrouche</p>
<p class="meta">Généré le {date.today():%d/%m/%Y} à partir des runs MLflow et des figures du dépôt
<code>github.com/fatimaamrch/PROJET_MLOps</code>. Modèle en production : <b>{champ_txt}</b>.</p>

<h2>1. Objectif et périmètre</h2>
<p>Construire un projet de machine learning complet et reproductible, exécutable entièrement en local, dans lequel
<b>MLflow trace l'intégralité des expériences</b> (paramètres, métriques, artefacts, données, modèles) et sert de
<b>Model Registry</b>, puis <b>déployer</b> le modèle retenu derrière une API. Le cas d'usage est la prédiction du
<i>churn</i> (résiliation) des clients d'un opérateur télécom : classification binaire sur données tabulaires mixtes.</p>

<div class="kpis">
  <div class="kpi"><div class="v">{ds["rows"]:,}</div><div class="l">clients après nettoyage (sur {ds["raw_rows"]:,})</div></div>
  <div class="kpi"><div class="v">{ds["n_num"] + ds["n_cat"]}</div><div class="l">features ({ds["n_num"]} numériques, {ds["n_cat"]} catégorielles)</div></div>
  <div class="kpi"><div class="v">{ds["churn_rate"]:.1%}</div><div class="l">taux de churn (classe positive)</div></div>
  <div class="kpi"><div class="v">{len(models)}</div><div class="l">versions dans le Model Registry</div></div>
  <div class="kpi"><div class="v">{fmt(champion["auc"] if champion else None)}</div><div class="l">ROC-AUC du champion (hold-out)</div></div>
</div>

<h2>2. Architecture du projet</h2>
<pre>
 CSV brut ──► src/data.py ──────────► data/processed/   (nettoyage, validation de schéma)
                    │
                    ▼
        src/pipeline.py + src/train.py ──► MLflow run (params, métriques, artefacts, dataset)
        ColumnTransformer + modèle           └─► Registry : TelcoChurnClassifier vN @champion
        GridSearchCV, autolog, log_model
                    │
                    ▼
        src/evaluate.py ───────────────► ROC / PR / confusion / rapport, loggés sur le même run
                    │
                    ▼
        app/main.py (FastAPI) ─────────► POST /predict   ← charge models:/…@champion
        Dockerfile / docker-compose         (fallback : artifacts/model.joblib)
</pre>
<table>
<tr><th>Composant</th><th>Implémentation</th></tr>
<tr><td>Configuration centralisée</td><td><code>configs/config.yaml</code> : chemins, cible, features, grilles d'hyperparamètres, CV, MLflow, seuil</td></tr>
<tr><td>Données</td><td><code>src/data.py</code> : téléchargement idempotent, nettoyage, validation de schéma (échec immédiat si colonne manquante)</td></tr>
<tr><td>Modélisation</td><td><code>src/pipeline.py</code> : <code>Pipeline(ColumnTransformer(imputer+scaler | imputer+one-hot), modèle)</code></td></tr>
<tr><td>Entraînement / tracking</td><td><code>src/train.py</code> : GridSearchCV (StratifiedKFold 5, ROC-AUC), autolog, dataset lineage, signature, deps épinglées, registry + alias</td></tr>
<tr><td>Évaluation</td><td><code>src/evaluate.py</code> : hold-out, ROC / PR / matrice de confusion, rapport de classification, prédictions CSV, loggés sur le run d'origine</td></tr>
<tr><td>Déploiement</td><td><code>app/main.py</code> (FastAPI + validation Pydantic), <code>Dockerfile</code>, <code>docker-compose.yml</code> (serveur MLflow + API)</td></tr>
<tr><td>Qualité / automatisation</td><td>22 tests pytest sur backend MLflow temporaire, ruff, Makefile, CI GitHub Actions (lint → tests → pipeline → build Docker)</td></tr>
</table>

<h2 class="pb">3. Exploration des données</h2>
<p>Détail dans <code>notebooks/eda.ipynb</code>. Points clés :</p>
<ul>
  <li><b>Qualité</b> : 11 valeurs <code>TotalCharges</code> vides, toutes pour des clients à <code>tenure = 0</code> (nouveaux clients, pas encore facturés) → converties en NaN et imputées par la médiane dans le pipeline ; 22 doublons exacts supprimés ; <code>SeniorCitizen</code> recodée en catégorielle ; identifiant client retiré.</li>
  <li><b>Cible</b> : déséquilibre modéré ({ds["churn_rate"]:.1%} de churn) → split et CV stratifiés, ROC-AUC comme métrique de sélection, <code>class_weight</code> dans la grille.</li>
  <li><b>Signal</b> : l'ancienneté (médiane 10 mois chez les churners contre 38), le type de contrat (43&nbsp;% de churn en mensuel contre 3&nbsp;% en 2 ans), la fibre optique, l'absence de services de sécurité/support et le paiement par chèque électronique sont les facteurs les plus discriminants.</li>
</ul>
<div class="row">
  <figure>{img(FIGURES / "target_distribution.png")}<figcaption>Distribution de la cible</figcaption></figure>
  <figure>{img(FIGURES / "correlation_heatmap.png")}<figcaption>Corrélations numériques</figcaption></figure>
</div>
<figure>{img(FIGURES / "numeric_boxplots.png")}<figcaption>Variables numériques par classe</figcaption></figure>
<figure>{img(FIGURES / "categorical_churn_rates.png", "92%")}<figcaption>Taux de churn par modalité (pointillé : taux moyen)</figcaption></figure>
<figure>{img(FIGURES / "tenure_contract_heatmap.png", "70%")}<figcaption>Le risque se concentre sur les clients récents en contrat mensuel</figcaption></figure>

<h2 class="pb">4. Entraînement et traçabilité MLflow</h2>
<p>Chaque exécution de <code>make train</code> crée un run MLflow parent (un run enfant par combinaison
d'hyperparamètres testée par GridSearchCV) et enregistre :</p>
<ul>
  <li><b>Paramètres</b> : tous les paramètres du pipeline et de la grille, plus les meilleurs (<code>best_model__C</code>…)</li>
  <li><b>Métriques</b> : score CV de chaque candidat, <code>best_cv_roc_auc</code>, métriques hold-out (<code>test_*</code>), métriques d'évaluation (<code>eval_*</code>)</li>
  <li><b>Artefacts</b> : <code>config.yaml</code>, <code>cv_results.csv</code>, courbes, matrice de confusion, rapport de classification, prédictions, et le <b>modèle</b> avec sa signature d'entrée/sortie, un exemple d'entrée et ses dépendances épinglées</li>
  <li><b>Données</b> : empreinte (digest) et schéma du jeu d'entraînement (<code>mlflow.log_input</code>)</li>
  <li><b>Registry</b> : nouvelle version de <code>{cfg["mlflow"]["registered_model_name"]}</code>, taguée, et alias <code>@{cfg["mlflow"]["alias"]}</code> déplacé dessus</li>
</ul>
<h3>Versions enregistrées dans le Model Registry</h3>
{models_table(models)}
<p>Les deux familles de modèles sont équivalentes en ROC-AUC (≈ 0,84). La forêt aléatoire avec
<code>class_weight="balanced"</code> détecte nettement plus de churners (rappel ≈ 0,73 contre ≈ 0,53) au prix d'une
précision plus faible : le choix du champion dépend du coût métier d'un churn manqué face à une action de rétention
inutile. Le seuil de décision (0,5 par défaut) est lui aussi un paramètre de configuration.</p>

<h2>5. Évaluation du modèle champion</h2>
<p>Hold-out de 20&nbsp;% (stratifié, seed 42), jamais vu pendant la recherche d'hyperparamètres.
ROC-AUC = <b>{fmt(ev.get("eval_roc_auc"))}</b>, accuracy = {fmt(ev.get("eval_accuracy"))},
précision = {fmt(ev.get("eval_precision"))}, rappel = {fmt(ev.get("eval_recall"))}, F1 = {fmt(ev.get("eval_f1"))}
(seuil {ev.get("eval_threshold", 0.5)}).</p>
<div class="row">
  <figure>{img(REPORTS / "roc_curve.png")}<figcaption>Courbe ROC</figcaption></figure>
  <figure>{img(REPORTS / "pr_curve.png")}<figcaption>Courbe précision-rappel</figcaption></figure>
  <figure>{img(REPORTS / "confusion_matrix.png")}<figcaption>Matrice de confusion</figcaption></figure>
</div>

<h2 class="pb">6. Déploiement</h2>
<p>L'API FastAPI (<code>make serve</code>, documentation Swagger sur <code>/docs</code>) charge
<code>models:/{cfg["mlflow"]["registered_model_name"]}@{cfg["mlflow"]["alias"]}</code> depuis le registry et,
à défaut, l'export local <code>artifacts/model.joblib</code> embarqué dans l'image Docker. Elle expose :</p>
<table>
<tr><th>Endpoint</th><th>Rôle</th></tr>
<tr><td><code>GET /health</code></td><td>liveness + source du modèle chargé</td></tr>
<tr><td><code>GET /model-info</code></td><td>version, run_id et métriques du modèle servi (traçabilité jusqu'au run MLflow)</td></tr>
<tr><td><code>POST /predict</code></td><td>un client (validation stricte des modalités et bornes) → probabilité, décision, seuil</td></tr>
<tr><td><code>POST /predict/batch</code></td><td>liste de clients</td></tr>
</table>
<pre>curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{{"tenure": 2, "MonthlyCharges": 85.7,
  "TotalCharges": 171.4, "gender": "Female", "SeniorCitizen": "0", "Partner": "No", "Dependents": "No",
  "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
  "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
  "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"}}'
→ {{"churn_probability": 0.77, "churn": true, "threshold": 0.5}}</pre>
<p>Deux modes de conteneurisation : <code>make docker-run</code> (image autonome avec healthcheck) et
<code>make docker-up</code> (docker compose : serveur MLflow avec son propre stockage + API qui charge le champion
depuis ce serveur). Promouvoir un nouveau modèle en production revient à déplacer l'alias dans l'UI MLflow, sans
modifier le code ni reconstruire l'image.</p>

<h2>7. Bonnes pratiques MLOps appliquées</h2>
<ul>
  <li><b>Reproductibilité</b> : configuration unique, seeds fixées, split déterministe, versions de dépendances épinglées (un modèle sérialisé avec scikit-learn 1.7.2 est servi avec scikit-learn 1.7.2 — un écart de version a été détecté et corrigé lors du premier build Docker).</li>
  <li><b>Traçabilité</b> : chaque version du registry pointe vers un run qui relie config, données (digest), hyperparamètres, métriques, figures et modèle ; l'API expose le <code>run_id</code> servi.</li>
  <li><b>Pas de training/serving skew</b> : preprocessing à l'intérieur du <code>Pipeline</code>, mêmes coercitions de types partagées entre entraînement et API, signature MLflow.</li>
  <li><b>Qualité continue</b> : tests unitaires et d'intégration (données, pipeline, tracking/registry sur backend temporaire, API), lint, CI qui rejoue le pipeline complet et construit l'image à chaque push.</li>
  <li><b>Séparation code / données / artefacts</b> : <code>data/</code>, <code>mlruns/</code>, <code>artifacts/</code>, <code>reports/</code> régénérables et hors git.</li>
  <li><b>Configuration 12-factor</b> : variables d'environnement prioritaires sur le YAML (<code>.env.example</code>).</li>
</ul>

<h2>8. Limites et pistes d'amélioration</h2>
<ul>
  <li>Pas de suivi de dérive (data/concept drift) en production : ajouter un job de monitoring comparant la distribution des requêtes au digest d'entraînement, et déclencher un ré-entraînement.</li>
  <li>Optimisation du seuil selon une fonction de coût métier plutôt que 0,5.</li>
  <li>Modèles supplémentaires (gradient boosting) et calibration des probabilités.</li>
  <li>Backend MLflow SQLite adapté au local ; en équipe, passer à PostgreSQL + stockage objet via le même <code>docker-compose</code>.</li>
</ul>

<h2>9. Reproduire</h2>
<pre>git clone https://github.com/fatimaamrch/PROJET_MLOps.git && cd PROJET_MLOps
make install && make pipeline     # data → train → evaluate, tout dans MLflow
make mlflow-ui                    # http://localhost:5000
make serve                        # http://localhost:8000/docs
make test && make lint            # 22 tests, ruff
make report                       # ce document (docs/rapport.pdf)</pre>
</body></html>"""


# ---------- PDF ----------


def html_to_pdf(html: Path, pdf: Path) -> bool:
    for cand in BROWSERS:
        exe = cand if Path(cand).exists() else shutil.which(cand)
        if not exe:
            continue
        cmd = [
            exe,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            html.resolve().as_uri(),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            if pdf.exists():
                return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"[report] {exe} failed: {exc}")
    return False


def main(no_pdf: bool = False) -> None:
    cfg = load_config()
    models = collect_models(cfg)
    ev = collect_eval(cfg)
    ds = collect_data_stats(cfg)
    DOCS.mkdir(exist_ok=True)
    html_path = DOCS / "rapport.html"
    html_path.write_text(render(cfg, models, ev, ds), encoding="utf-8")
    print(f"[report] HTML written to {html_path} ({len(models)} model versions)")
    if no_pdf:
        return
    pdf_path = DOCS / "rapport.pdf"
    if html_to_pdf(html_path, pdf_path):
        print(f"[report] PDF written to {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
    else:
        print("[report] no Chrome/Edge found: open docs/rapport.html in a browser and print to PDF")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdf", action="store_true")
    main(parser.parse_args().no_pdf)
