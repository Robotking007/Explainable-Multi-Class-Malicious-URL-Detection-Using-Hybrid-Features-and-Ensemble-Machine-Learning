# Explainable-Multi-Class-Malicious-URL-Detection-Using-Hybrid-Features-and-Ensemble-Machine-Learning
This project develops an explainable ML-based system to detect malicious URLs as benign, phishing, malware, or defacement. It combines lexical, TF-IDF/N-gram, Host/WHOIS and structural features with SMOTE and a Voting Classifier. Suspicious URLs undergo dynamic analysis, with risk scoring, XAI, and real-time web deployment.

It is being extended into a two-branch platform that also detects malware from dynamic runtime/behavioral features, with both branches sharing a common SHAP/XAI explanation contract, per the project synopsis in `Docs/` (or as supplied separately).

## Repository layout

```
Dataset/                          datasets used to train both branches (not all tracked in git; see below)
src/
  url_pipeline/
    features.py                   lexical, host, content-keyword, structural, statistical + char n-gram TF-IDF features
    train.py                      feature selection (RF importance) + SMOTE + Voting Classifier (RF+XGBoost+LR+NB)
    explain.py                    SHAP-based per-URL prediction explanation (Module D)
    dynamic_escalation.py          Module C: headless-browser redirect/DOM/script analysis for low-confidence URLs
  malware_pipeline/
    train_dynamic_baseline.py     dynamic memory-forensics feature baseline (RF+XGBoost) + SHAP, binary and family-level
models/                           trained model/encoder/vectorizer artifacts (.joblib, .json)
reports/                          run logs and metrics (.json) for each branch
```

## Status

### Module B — URL static-feature Voting Classifier (`src/url_pipeline/`)
Trained on `Dataset/malicious_phish.csv` (651,199 URLs; benign / phishing / malware / defacement).
Pipeline: lexical + host + content-keyword + structural + statistical features, plus a char n-gram TF-IDF
block, fed through Random-Forest-importance feature selection, SMOTE balancing on the train split, and a
soft-voting ensemble (Random Forest + XGBoost + Logistic Regression + Naive Bayes). Live WHOIS lookups are
intentionally excluded from the inline feature extractor since a per-URL network query is infeasible at this
scale; only structural/lexical host signals are used.

Full run on all 651,199 URLs (80/20 stratified split, 130,239-URL held-out test set) — **first full run, before
the scheme-normalization fix described below**:

| Metric | Value |
|---|---|
| Accuracy | 96.2% |
| Macro Precision | 94.5% |
| Macro Recall | 94.5% |
| Macro F1 | 94.4% |
| Macro ROC-AUC | 0.993 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Benign | 0.98 | 0.98 | 0.98 |
| Defacement | 0.96 | 1.00 | 0.98 |
| Malware | 0.95 | 0.94 | 0.95 |
| Phishing | 0.89 | 0.86 | 0.88 |

Phishing was the weakest class (13.8% false-negative rate). These specific numbers are being superseded by a
retrain with the fix below; check `reports/url_module_b_metrics.json` for the current numbers and this section
for the next update.

**Bugs found and fixed while validating this model (all three matter for reproducing/rerunning training):**

1. **Scheme-prefix leak (correctness bug).** SHAP explanations on the first trained model showed
   `https://www.google.com` classified as phishing with 97% confidence. Root cause: in
   `Dataset/malicious_phish.csv`, only ~8% of benign rows carry an explicit `http(s)://` prefix, versus ~96-100%
   of malware/defacement rows — an artifact of how each class was collected, not a real phishing signal. The
   model was learning "has a scheme prefix" as a malicious proxy. Fixed in `features.py::normalize_url()`,
   which strips any caller-supplied scheme before feature extraction, consistently at train and inference time;
   the now-meaningless `https_flag` feature was removed. A second, smaller effect remains and is a genuine
   dataset-composition limitation rather than a code bug: bare root domains (no path) are ~23% of phishing URLs
   here but ~0.07% of benign ones, so top-level domains outside this dataset's benign sample (e.g. brand-name
   root domains) can still get flagged more than they should.
2. **`OSError: No space left on device`** during `VotingClassifier.fit()` — scikit-learn's parallel fit
   memmaps large arrays to the OS temp folder, and the C: drive here runs at ~97-99% capacity. Fix: point temp
   storage at a drive with room before training:
   ```
   set TMP=D:\path\to\some\tmp\folder
   set TEMP=D:\path\to\some\tmp\folder
   set JOBLIB_TEMP_FOLDER=D:\path\to\some\tmp\folder
   ```
3. **`MemoryError`** during `VotingClassifier.fit()` on the full, SMOTE-balanced ~1.37M-row training set —
   `VotingClassifier(n_jobs=-1)` forks one OS process per estimator, and each fork gets its own pickled copy of
   the full training matrix; on a 16GB-RAM machine that reliably exhausts memory. Fixed by fitting the four
   estimators sequentially (`n_jobs=None`) in `train.py`; each estimator still parallelizes internally.

**GPU support:** `train.py` and `train_dynamic_baseline.py` both auto-detect an NVIDIA GPU and run XGBoost on
CUDA when available (falls back to CPU otherwise). Random Forest / Logistic Regression / Naive Bayes have no
GPU backend in scikit-learn without RAPIDS cuML, so those stay CPU-bound; XGBoost training time dropped
roughly 5x on an RTX 3050 in testing.

Run it:
```
python src/url_pipeline/train.py --sample-size 0 --top-k 30 --ngram-features 60
python src/url_pipeline/explain.py "http://paypal-secure-login.tk/verify/account"
```

### Module A — malware dynamic-feature baseline (`src/malware_pipeline/`)
The malware datasets on disk (`Obfuscated-MalMem2022.parquet`, `bodmas.parquet`) are pre-aggregated feature
tables rather than raw sandbox event logs, so the full behavior-graph/GraphSAGE design from the synopsis
can't be built from them directly — that needs either live CAPE/Cuckoo sandbox access or a raw API-call-log
corpus, neither of which is present. As an interim, real deliverable, `train_dynamic_baseline.py` trains a
Random Forest + XGBoost baseline directly on the CIC-MalMem-2022 dynamic memory-forensics features
(process/DLL/handle/injection/service statistics) for two tasks:

| Task | Accuracy | Macro F1 | Macro ROC-AUC |
|---|---|---|---|
| Benign vs. Malware (binary) | 100% | 1.00 | 1.00 |
| Malware family — Ransomware/Spyware/Trojan/Benign (4-class) | 87.8% | 0.81 | 0.97 |

SHAP explanations for both tasks point to the same top signals: number of running services
(`svcscan.nservices`), shared process services, average handles per process, and kernel driver count — i.e.
malware families in this corpus tend to install/hook services and hold unusual handle counts. Full metrics:
`reports/malware_module_a_dynamic_baseline_metrics.json`.

Run it:
```
python src/malware_pipeline/train_dynamic_baseline.py
```

### Module C — URL dynamic/sandbox escalation path (`src/url_pipeline/dynamic_escalation.py`)
Runs a suspicious/low-confidence URL through a headless, isolated Chromium context (Playwright) and captures
the Table 6.2 dynamic signals: redirect chain and domain changes, login-form presence and cross-domain form
submission, external-resource/script counts, inline-script obfuscation score (char entropy), popup/dialog
triggers on load, page-title/brand mismatch, a screenshot perceptual hash, and whether TLS was used. Safety
controls: hard per-navigation timeout, blocked downloads, no granted permissions (camera/mic/geolocation), no
execution of anything downloaded. `should_escalate()` implements the confidence-gate rule (escalate when the
static model's confidence is below threshold, or the class is phishing/malware below 90% confidence).

Verified working end-to-end: `google.com` → `www.google.com` correctly reports `redirect_count: 2` and
`domain_changed: true`; static pages (Wikipedia, a live 404) correctly report `redirect_count: 0`. One bug was
found and fixed during this verification: the initial implementation counted every network response
(images/scripts/stylesheets), not just top-level document navigations, which inflated `redirect_count` (e.g.
4 for a Wikipedia load with no real redirect). Fixed by filtering to `resource_type == "document"` responses
on the page's main frame only.

Setup (Chromium is installed to `ms-playwright/` on the D: drive here, not the default `C:\Users\<user>\AppData\Local\ms-playwright`,
since C: is at ~97-99% capacity):
```
pip install playwright imagehash
set PLAYWRIGHT_BROWSERS_PATH=D:\path\to\ms-playwright
python -m playwright install chromium
```
Run it (needs `PLAYWRIGHT_BROWSERS_PATH` set in the shell if installed non-default as above):
```
python src/url_pipeline/dynamic_escalation.py "http://example-suspicious.com"
```
Note: this script does not itself provide network isolation (e.g. a container with restricted egress) — that
should wrap it in production, per Section 6.2.1 step 8 of the synopsis.

### Module A' — static PE-feature malware baseline (`src/malware_pipeline/train_static_baseline.py`)
Complements the dynamic memory-forensics baseline above with a static-feature branch, combining two datasets
that share the identical 2,381-dimension EMBER-style feature schema (BODMAS was built on EMBER's own feature
extractor, specifically for cross-compatibility):
- `Dataset/bodmas.parquet` (134,428 rows, binary Label)
- `Dataset/train_ember_2018_v2_features.parquet` (799,912 rows; unlabeled `Label == -1` rows excluded)

`Dataset/test_ember_2018_v2_features.parquet` (199,956 rows, fully labeled) is held out as a genuine
external/unseen validation set — a different vintage/distribution than the training pool, matching the
external-validation requirement in Section 9 of the synopsis rather than just another random split of the
same pool. Tree depth is capped (`max_depth=16`) up front, learning directly from the URL branch's SHAP-speed
incident below rather than repeating it. See `reports/malware_module_a_static_baseline_metrics.json` for
results once trained (`python src/malware_pipeline/train_static_baseline.py`).

### Web application (`src/api/main.py`, `frontend/code.html`)
A FastAPI backend serves the trained models over HTTP and hosts the dashboard itself (`GET /` returns
`frontend/code.html`), so opening `http://127.0.0.1:8000/` in a browser gives a same-origin page with no CORS setup
needed. The frontend was originally a Google AI Studio ("Stitch") export whose "scan" button, probability
bars, and SHAP list were pure keyword-matched/hardcoded demo data with no backend behind them at all (and
whose sample SHAP feature names, like `absence_of_valid_whois_record`, didn't even match this project's real
feature names). It has since been rebuilt to call this real API for every action except the four small
"quick demo sample" buttons, which are intentionally kept hardcoded and labeled as illustrative.

The dashboard has two tabs: **Threat Analyzer** (everything above, all real/API-backed except the four labeled
demo-sample buttons) and **3D Visualizer** (an illustrative, decorative Three.js rendering of the four URL
classes as clustered points in a conceptual 3D feature space, adapted from `frontend/reference/3.html`/`4.html`
in this repo, kept there as the original design exports this was adapted from; the animation is clearly labeled as not driven by live
data, and its GPU work pauses whenever that tab isn't visible).

Endpoints:
```
GET  /                      the dashboard (frontend/code.html)
GET  /api/health            liveness + which models are loaded
GET  /api/metrics           real, current accuracy/ROC-AUC etc., read live from reports/*.json on every call
                             (never hardcoded into the page -- the original mockup's "89.8%" KPI strip had
                             already gone stale by the time it was checked against the real trained model)
POST /api/url/predict       {url} -> Module B prediction + SHAP explanation
POST /api/url/escalate      {url} -> Module C dynamic/sandbox analysis
GET  /api/malware/sample    ?task=binary|family -> Module A prediction + SHAP on one real held-out row
```
Run it:
```
set PLAYWRIGHT_BROWSERS_PATH=D:\path\to\ms-playwright
uvicorn src.api.main:app --port 8000
```
then open `http://127.0.0.1:8000/`. The dashboard's "Recent Scans" table only logs genuine API responses
(URL scans and Module A memory samples) — never the hardcoded demo-sample buttons — so it always reflects
real model output, unlike the original mockup's live-stream section, which cycled fabricated confidence
scores from a static pool and was removed for exactly that reason.

**Bug found and fixed while wiring this up:** the first version of `/api/url/predict` took 90+ seconds and
multiple GB per request, which looked exactly like a hung server. Root cause: the production Random Forest
has 300 unconstrained-depth trees (~depth 60-72, ~54,000 leaves each — it was tuned for accuracy on ~1.4M
rows, not for cheap explanation), and exact TreeSHAP over a forest that shape is genuinely that slow. Fixed
with `src/url_pipeline/train_shap_surrogate.py`: a separate, shallow (`max_depth=8`, ~100 trees) surrogate RF
trained on the same feature space, used only to compute fast SHAP attributions at serving time. Predictions
and confidence in the API response still come from the full production ensemble; only "top contributing
factors" come from the surrogate, and the response says so explicitly (`explanation_model` field). Retrain it
after retraining the main model: `python src/url_pipeline/train_shap_surrogate.py --sample-size 40000`.

### Combined URL training corpus (`src/url_pipeline/build_combined_dataset.py`)
Merges four URL sources into one deduplicated training set: `malicious_phish.csv` (original, 651,191 rows),
`MUD_malicious_urls_2026_V2.csv` (newer/larger, 632,844 rows), `verified_online.csv` (PhishTank's live-verified
feed, phishing-only, 76,403 rows), and `merged_urls_dataset.csv` (518,293 usable rows after dropping 3.37M
out-of-taxonomy "adult"-labeled rows). Of ~1.88M raw rows before dedup, ~523k are exact same-URL-same-label
duplicates across sources (safely dropped) and a separate ~4,100 URLs (~0.4%) have a genuine label
disagreement across sources — those are resolved by explicit source-priority order (verified_online > MUD >
malicious_phish > merged_urls_dataset, most-to-least authoritative), not dropped, so no unique URL is lost to
resolve a conflict. Output: `Dataset/combined_urls.csv`, currently 1,052,060 rows (benign 586,855 / phishing
239,452 / defacement 161,626 / malware 64,127 — malware more than doubled versus the original 651k-row corpus,
directly addressing that class's small size).

### Not yet started
- Fusion layer correlating malware and URL risk scores into a single incident record.
- `Dataset/API_Functions.csv` / `PE_Header.csv` / `DLLs_Imported.csv` / `PE_Section.csv` (1.3GB+ combined):
  appear to be components of another malware feature dataset needing a join across files — not yet
  investigated.
- `Dataset/Malware_and_benign_recognition.csv` / `benign.csv` / `malware.csv` (~1,150 rows) and
  `dataset_malwares.csv`/`dataset_test.csv`: small, each using its own PE-header feature schema incompatible
  with the EMBER/BODMAS 2,381-feature set above, so not merged into Module A'. Could become a small separate
  baseline later.

### Explicitly out of scope
- `Dataset/diestilbert_sft_train_350_175_250.csv` / `..._test_...csv` (3.6GB combined): raw disassembly text
  for fine-tuning a transformer (DistilBERT-style) model on malware detection from code text. This is a
  fundamentally different task from the classical-ML ensembles in this repo (needs a tokenizer, a training
  loop, and meaningfully more GPU VRAM than the 6GB available here) and was intentionally not attempted
  without explicit confirmation.
- SOREL-20M (`s3://sorel-20m/09-DEC-2020/processed-data/`): confirmed public and reachable, but its
  `pe_metadata` shard alone is 515GB (590GB+ total), far beyond what fits on this machine. Not downloaded.
  `train_ember_2018_v2_features.parquet` / `test_ember_2018_v2_features.parquet` (already used in Module A'
  above) serve the same static-feature purpose at a fraction of the size.
- No chatbot is built here, per an earlier explicit instruction not to add one.
