# Explainable Multi-Class Malicious URL Detection Using Hybrid Features and Ensemble Machine Learning

An explainable ML system with four trained modules: a URL classifier (benign / phishing /
defacement / malware) and three malware classifiers (static PE-features, dynamic
memory-forensics, and API-call-sequence family classification). Every prediction comes with a
SHAP-based explanation. All heavy training runs on Kaggle's cloud compute, not locally — see
[Kaggle infrastructure](#kaggle-infrastructure--compute) below for why and how.

## Repository layout

```
Dataset/                          training data (gitignored — multi-GB, never committed; see below)
models/                           trained model/encoder/vectorizer artifacts (.joblib, .json) — gitignored,
                                   auto-downloaded from HF Hub on demand by model_registry.py
reports/                          run logs and metrics (.json) for each module

src/
  url_pipeline/
    features.py                    lexical, host, content-keyword, structural, statistical
                                    features + char n-gram TF-IDF (chunked for memory safety at scale)
    whois_features.py              live WHOIS lookup (domain age/registration/privacy) — single-domain,
                                    used for ablation studies + live API predictions, not bulk training
    asn_features.py                live DNS + Team Cymru ASN/IP-prefix lookup — same scope as WHOIS
    whois_bulk_enrich.py           local checkpointed bulk WHOIS enrichment (superseded by the Kaggle chain)
    asn_bulk_enrich.py             local checkpointed bulk ASN enrichment (DNS + batched Cymru lookups)
    build_combined_dataset.py      merges ~20 URL sources into Dataset/combined_urls.csv
    train.py                       local reference copy of the training pipeline (mirrors the Kaggle kernel;
                                    NOT used for actual training — see Kaggle infrastructure below)
    explain.py                     SHAP-based per-URL prediction + live WHOIS/ASN host-intel lookup (Module D)
    dynamic_escalation.py          Module C: headless-browser redirect/DOM/script analysis for low-confidence URLs
    train_shap_surrogate.py        fast shallow-RF surrogate used only for SHAP speed (see below)
    run_whois_ablation.py          reproduces a WHOIS-features ablation study at a bounded sample size

  malware_pipeline/
    train_dynamic_baseline.py      Module A: CIC-MalMem-2022 memory-forensics RF+XGBoost baseline
    train_static_baseline.py       Module A': BODMAS+EMBER static PE-feature RF+XGBoost baseline
    train_api_sequence_baseline.py Module A'': MalAPI2019 API-call-sequence family classifier
    explain.py / explain_static.py SHAP explanation for the dynamic / static malware modules

  api/
    main.py                        FastAPI backend serving all four modules + the dashboard
    chatbot.py                     Gemini-backed scan explainer (hidden behind a feature flag)

  model_registry.py                downloads a model from HF Hub if missing locally, no-op otherwise
  fetch_abusech_data.py            pulls ThreatFox + URLhaus bulk exports (authenticated)

  kaggle_*.py                      Kaggle kernel push/watch/chain scripts — see Kaggle infrastructure below

frontend/
  code.html                        the dashboard (Threat Analyzer + 3D Visualizer tabs, Host Intelligence panel)
```

## Module status

### Module B — URL classifier (`src/url_pipeline/`)

**Training data:** `Dataset/combined_urls.csv`, merged from ~20 public sources (Kaggle datasets,
Hugging Face datasets, PhishTank's live feed, abuse.ch ThreatFox/URLhaus) by
`build_combined_dataset.py`, with explicit source-priority resolution for the small fraction of
URLs that appear in more than one source under a different label (never silently dropped).

**Current size: 20,946,068 rows** — benign 11,292,992 / phishing 9,264,267 / defacement 201,997 /
malware 186,812. Cross-source label disagreement (the same URL appearing under a different class in
two sources) affects ~2.16% of unique URLs — resolved by the priority order below, never dropped.

<details>
<summary><strong>Full list of merged sources (click to expand)</strong></summary>

Merged in explicit priority order (highest first — wins on label disagreement):

1. `verified_online.csv` — PhishTank's live-verified feed (phishing-only, human-verified)
2. `file.csv.txt` — abuse.ch URLhaus dump (malware-only, curated threat feed)
3. `MUD_malicious_urls_2026_V2.csv` — newer (2026), large 4-class corpus
4. `malicious_phish.csv` — original corpus (2021/2022 vintage), 4-class
5. `merged_urls_dataset.csv` — 3.9M rows total, 3.37M out-of-taxonomy "adult" rows dropped
6. `PhishVN.../dataset_url.csv` — PhishVN v3.1.0 (Vietnamese phishing dataset)
7. `zhengxian_12M_urls.csv` — 12M-row Kaggle set (legitimate/phishing)
8. `somanshumahajan_6M_urls.csv` — 6.1M-row Kaggle set (0=benign/1=phishing, label polarity verified by URL sampling)
9. `kmack_phishing_urls.csv` — HF dataset, 708,820 rows (0=benign/1=phishing, verified)
10. `Malicious URL v3.csv` / `Malicious URL Detection.csv` — direct 4-class match
11. `ASA_Phishing_URL_Dataset.csv`, `URL dataset.csv`, `StealthPhisher2025.csv` — legitimate/phishing
12. `Phishing URLs.csv` — phishing-only
13. `Dataset.csv`, `DomainGuard-50K.csv`, `dataset2.csv` — binary 0=benign/1=phishing (verified)
14. `url_features_extracted1.csv`, `PhiUSIIL_Phishing_URL_Dataset.csv` — binary, **inverted** polarity
    (1=benign/0=phishing) — verified by sampling; the opposite convention from #13's sources, so
    getting this backwards would have silently mislabeled hundreds of thousands of rows
15. `balanced_urls.csv`, `Dataset I.csv`, `Dataset II.csv` — ambiguous "benign vs malicious" sources;
    only the benign side is usable (the malicious side doesn't say which of phishing/defacement/
    malware it is, so guessing would fabricate a label)
16. `phreshphish_urls.csv` — HF `phreshphish/phreshphish`, 666,315 rows collected 2025-2026 (fresh,
    out-of-time real phishing/benign URL+HTML pairs; only url/label used, HTML not downloaded)
17. `naren1704_urls.csv` — HF `Naren1704/phishing-dataset`'s `urls.json` config, 835,697 rows (a
    compilation of a Kaggle 800K-URL set plus JPCERT/PhishTank/OpenPhish-derived sources; heavy
    overlap with sources above is expected and harmless, resolved by the same dedup step)
18. `threatfox_full.csv` / refreshed `file.csv.txt` — abuse.ch ThreatFox bulk export (authenticated,
    `fetch_abusech_data.py`), malware-only
19. `satyammishra_1.5M_benign_domains.csv` — capped sample (from 13.6M available) of real registered
    business websites, no label column, safely all-benign by construction

One candidate source (`Phishing Website Detection Dataset/`, 6.5M headerless plaintext URLs) was
investigated and deliberately **excluded**: sampling its "malign" file found 14,641 lines referencing
amazon.com/google.com/wikipedia.org/microsoft.com plus adult-content URLs in both classes — strong
evidence it's a mislabeled adult-content classifier, not a phishing/malware corpus.

</details>

**Pipeline:** lexical + host + content-keyword + structural + statistical features (`features.py`),
plus a char n-gram TF-IDF block (100 features, fit on train only), fed through Random-Forest
feature-importance selection (top-30 engineered features), capped/targeted SMOTE (minority classes
oversampled to 1.2M, not full parity with benign — full parity at this scale would need ~40M+ mostly
synthetic rows), `sample_weight`-balanced soft-voting ensemble (Random Forest + XGBoost + Logistic
Regression + Naive Bayes).

<details>
<summary><strong>Full engineered feature list, 45 features (click to expand)</strong></summary>

All computed directly from the URL string (`extract_features_row()` in `features.py`) — no live
page fetch or network call, so this block is cheap enough to run inline at full dataset scale
(unlike the WHOIS/ASN block below).

**Lexical (11):** `url_length`, `hostname_length`, `path_length`, `digit_ratio`, `uppercase_ratio`,
`ip_in_url_flag`, and 8 special-character counts (`count_at`, `count_qmark`, `count_hyphen`,
`count_eq`, `count_dot`, `count_slash`, `count_pct`, `count_amp`, `count_colon`) — 9 counts total.
No `https_flag`: `normalize_url()` strips any scheme before feature extraction (see below), so
scheme presence isn't observable and would otherwise be a constant, uninformative column.

**Host-based (10):** `subdomain_count`, `domain_length`, `tld_length`, `suspicious_tld_flag` (checked
against a 26-entry list of TLDs commonly abused by cheap/free registrars — `.tk`, `.ml`, `.ga`,
`.cf`, `.xyz`, `.top`, `.click`, etc.), `hyphen_in_domain_flag`, `digit_in_domain_flag`,
`domain_entropy` (Shannon entropy of the registered domain), `subdomain_length`,
`domain_token_count`, `brand_in_domain_flag` / `brand_in_subdomain_flag` (checked against a
12-brand keyword list: paypal, amazon, apple, google, microsoft, facebook, netflix, bank,
instagram, whatsapp, outlook, icloud).

**Content-keyword (4):** `phish_keyword_count` (login/signin/verify/secure/confirm/banking/
password/etc., 13 keywords), `brand_keyword_count`, `action_keyword_count` (download/install/
click/free/gift/bonus/win), `sensitive_keyword_count` (ssn/creditcard/cvv/otp/pin/wallet) — all
computed on the URL string itself, no page content is fetched.

**Structural (7):** `path_depth`, `param_count`, `has_fragment_flag`, `double_slash_in_path_flag`,
`query_length`, `path_entropy`, `query_entropy`.

**Statistical (8):** `url_entropy`, `compression_ratio` (zlib-compressed length ÷ raw length —
low-entropy/repetitive URLs compress more), `char_diversity`, `token_count`, `avg_token_length`,
`max_token_length`, `unique_token_ratio`, `special_char_ratio` (URL split on `/-_.?=&%@:`
separators for the token-level measures).

**Char n-gram TF-IDF block (100 features, separate from the 45 above):** char-level 2-3-grams,
`TfidfVectorizer(analyzer="char", ngram_range=(2,3), max_features=100)`, fit on the train split
only (never the test split, to avoid leakage) via `build_ngram_features()`.

Of these 45 engineered features, only the **top 30 by Random-Forest importance** are kept for the
final model (`TOP_K = 30` in the training scripts) — ranked using a shallow (`max_depth=20`),
100-tree RF on a 2M-row subsample, not the full training set (see
[OOM crash history](#oom-crash-history-url-module-training), fix #4, for why).

**Supplementary, NOT trained-model features — live lookups only:**
- **WHOIS** (`whois_features.py`, 5 features): `whois_available`, `domain_age_days`,
  `registration_length_days`, `recent_domain_flag` (registered <90 days ago),
  `whois_private_flag` (registrant identity redacted/private).
- **ASN / IP-prefix** (`asn_features.py`, 5 features): `asn_available`, `asn` (numeric Autonomous
  System Number), `bgp_prefix_length`, `registry` (which RIR allocated the block — arin/ripencc/
  apnic/lacnic/afrinic), `asn_country_code`, plus `resolved_ip` for display. Two-step lookup: DNS
  A-record resolution, then IP→ASN mapping via Team Cymru's free bulk-whois service
  (`whois.cymru.com:43`) — no API key required.

</details>

**Status: retraining in progress on Kaggle** (see below) with a corrected pipeline. The last
completed run (84.8% accuracy, on an earlier/smaller version of the dataset, before four rounds of
memory fixes — see [OOM crash history](#oom-crash-history-url-module-training)) is superseded and
kept in `reports/url_module_b_metrics.json` only as a historical reference — check that file for
whatever the most recent completed run reports.

WHOIS/ASN host-intelligence features are **not** part of the trained model (both are live,
per-domain network lookups, infeasible inline at 20M+-row scale) — they're a separate, supplementary
layer: shown alongside predictions in the API/frontend, and available for smaller-scale ablation
studies. See [Host intelligence enrichment](#host-intelligence-enrichment-whois--asn) below.

Run it (local reference only — actual training runs on Kaggle):
```
python src/url_pipeline/train.py --sample-size 0 --top-k 30 --ngram-features 100
python src/url_pipeline/explain.py "http://paypal-secure-login.tk/verify/account"
```

### Module A — dynamic (memory-forensics) malware classifier (`src/malware_pipeline/train_dynamic_baseline.py`)

Trained on CIC-MalMem-2022's memory-forensics features (process/DLL/handle/injection/service
statistics extracted via Volatility from a memory dump — not live sandbox traces).

| Task | Accuracy | Macro F1 | Macro ROC-AUC |
|---|---|---|---|
| Benign vs. Malware (binary) | 100% | 1.00 | 1.00 |
| Malware family — Ransomware/Spyware/Trojan/Benign (4-class) | 87.8% | 0.815 | 0.973 |

Ransomware and Trojan are the weakest pair (FNR ~26-28%, meaningful mutual confusion); Spyware is
best-separated. Top SHAP features: `svcscan.nservices`, `svcscan.shared_process_services`,
`handles.avg_handles_per_proc`, `svcscan.kernel_drivers` — malware families in this corpus tend to
install/hook services and hold unusual handle counts. Full metrics:
`reports/malware_module_a_dynamic_baseline_metrics.json`.

A richer real-behavioral dataset (Avast-CTU CAPE, 48,976 genuine CAPEv2 sandbox reports with
registry/network/process-tree data and 10 real family labels) was identified as a strong upgrade
path but requires a new feature-extraction pipeline (parsing raw JSON reports, not a ready CSV) —
deferred, not started.

Run it: `python src/malware_pipeline/train_dynamic_baseline.py`

### Module A' — static (PE-feature) malware classifier (`src/malware_pipeline/train_static_baseline.py`)

Trained on BODMAS + EMBER-2018 (identical 2,381-dimension EMBER-style feature schema — BODMAS was
built on EMBER's own extractor specifically for cross-compatibility), with EMBER's own held-out test
set kept as a genuine external/unseen validation split (different vintage than the training pool).

| Split | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Internal holdout (146,870 rows) | 98.1% | 0.980 | 0.998 |
| External EMBER test set (199,956 rows) | 96.6% | 0.966 | 0.994 |

Tree depth capped (`max_depth=16`) from the start, applying the lesson from the URL module's SHAP
serving-speed incident before it could recur here. Full metrics:
`reports/malware_module_a_static_baseline_metrics.json`.

Run it: `python src/malware_pipeline/train_static_baseline.py`

### Module A'' — API-call-sequence malware family classifier (`src/malware_pipeline/train_api_sequence_baseline.py`)

Trained on MalAPI2019 (7,107 samples, 8 malware families — Adware/Backdoor/Downloader/Dropper/
Spyware/Trojan/Virus/Worms, no benign class). Word-level TF-IDF (1-3 grams, 500 features) over
space-joined API-call-ID sequences, RF + XGBoost.

| Metric | Value |
|---|---|
| Accuracy | 67.3% |
| Macro Precision | 69.5% |
| Macro Recall | 68.3% |
| Macro F1 | 68.9% |
| Macro ROC-AUC | 0.918 |

**This is the weakest module.** Trojan (FNR 50%) and Spyware (FNR 44%) are especially poorly
separated. The Oliveira Kaggle dataset (42,797 malware sequences, same format as MalAPI2019) was
evaluated as a fix but rejected — it only has binary malware/benign labels, no family breakdown, so
it can't supply ground truth for this 8-class task. The Avast-CTU CAPE dataset (see Module A above)
would also help here, since it includes real family-labeled API sequences at larger scale. Full
metrics: `reports/malware_module_a_api_sequence_metrics.json`.

Run it: `python src/malware_pipeline/train_api_sequence_baseline.py`

### Module C — URL dynamic/sandbox escalation (`src/url_pipeline/dynamic_escalation.py`)

Runs a suspicious/low-confidence URL through a headless, isolated Chromium context (Playwright) and
captures redirect chain and domain changes, login-form presence and cross-domain form submission,
external-resource/script counts, inline-script obfuscation score, popup/dialog triggers, page-title/
brand mismatch, a screenshot perceptual hash, and TLS usage. Safety controls: hard per-navigation
timeout, blocked downloads, no granted permissions, no execution of anything downloaded.
`should_escalate()` gates on prediction confidence (escalate when confidence is below threshold, or
the class is phishing/malware below 90% confidence).

Setup:
```
pip install playwright imagehash
set PLAYWRIGHT_BROWSERS_PATH=D:\path\to\ms-playwright
python -m playwright install chromium
```
Run it: `python src/url_pipeline/dynamic_escalation.py "http://example-suspicious.com"`

### Host intelligence enrichment (WHOIS + ASN)

Two supplementary, live-lookup feature families — domain registration data (WHOIS) and hosting
infrastructure data (ASN/IP-prefix, from Ma et al.'s malicious-URL research) — shown in the frontend
alongside predictions, never part of the trained model itself (both require per-domain network
lookups, infeasible inline at 20M+-row training scale).

**Bulk coverage status** (against the current dataset's 11,744,984 unique registered domains):
- **ASN**: 100% coverage (all 11.7M domains attempted), 48.8% resolved to a real ASN — the rest are
  domains that don't resolve via DNS (dead/parked/typosquat domains, expected at this scale). Fast
  because Team Cymru's bulk-whois service (`whois.cymru.com:43`) has no per-registrar rate-limit
  fragmentation — one batched TCP query covers up to ~1000 IPs.
- **WHOIS**: ~8.5% coverage (997,994/11.7M domains). Much slower — every registrar rate-limits
  independently, so full coverage at this domain count isn't practically achievable the way ASN's
  was; chained across multiple Kaggle sessions, still partial.

### Web application (`src/api/main.py`, `frontend/code.html`)

FastAPI backend serving all four trained modules over HTTP, hosting the dashboard itself
(`GET /` returns `frontend/code.html`) so opening `http://127.0.0.1:8000/` needs no CORS setup.

Endpoints:
```
GET  /                      the dashboard (frontend/code.html)
GET  /api/health            liveness + which models are loaded
GET  /api/metrics           real, current accuracy/ROC-AUC etc., read live from reports/*.json
POST /api/url/predict       {url} -> Module B prediction + SHAP + live WHOIS/ASN host-intel lookup
POST /api/url/escalate      {url} -> Module C dynamic/sandbox analysis
GET  /api/malware/sample           ?task=binary|family -> Module A prediction + SHAP on a real held-out row
GET  /api/malware/static_sample    -> Module A' prediction + SHAP on a real EMBER-test row
POST /api/chat              scan-aware chatbot (Gemini-backed, hidden behind a feature flag)
```
Run it:
```
set PLAYWRIGHT_BROWSERS_PATH=D:\path\to\ms-playwright
uvicorn src.api.main:app --port 8000
```
then open `http://127.0.0.1:8000/`. The dashboard's **Threat Analyzer** tab is fully real/API-backed
(the four "quick demo sample" buttons are the only intentionally-hardcoded illustrative exception);
**3D Visualizer** is a decorative, clearly-labeled Three.js rendering, not live data. The **Host
Intelligence** panel shows the live WHOIS/ASN lookup for the last real URL scan.

## Kaggle infrastructure & compute

Every training run for all four modules happens on Kaggle's cloud compute, not this machine — both
because the datasets (20M+ rows) and models (up to 14.6GB) exceed what this laptop can reliably hold
in memory, and per explicit project decision so training survives the laptop sleeping/closing.
`src/kaggle_*.py` scripts handle uploading datasets, pushing kernels, and polling/downloading results.

**Two Kaggle accounts are in play**, because Kaggle enforces a hard 30-hour/week GPU quota per
account: the primary account (`siddudavant`) hit that cap mid-project, so a second account
(`siddhantudavant`) was set up specifically for URL Module B training (the GPU-heavy module) to keep
working while the primary account's quota resets weekly. WHOIS/ASN/static/dynamic/API-sequence
kernels are CPU-only and don't compete for the GPU quota at all.

### OOM crash history (URL module training)

The URL training pipeline crashed from out-of-memory four separate times while scaling up to the
current ~21M-row dataset, each a genuinely different root cause (fixed in order, each verified before
moving to the next):

1. **Redundant data copies.** The original code built static features, n-gram features, a
   concatenated "full" copy, and a selected-columns "sel" copy — for both train and test — without
   freeing any precursor until the very end. Up to ~7 large, nearly-redundant copies alive
   simultaneously (estimated ~35-40GB peak). Fixed by processing train/test sequentially and deleting
   each precursor immediately after its last use.
2. **float64 densification.** `TfidfVectorizer`'s sparse output defaults to float64; densifying a
   150-feature n-gram block at ~16.7M rows needs ~19GB in float64 alone. Fixed by casting to float32
   *before* densifying (not after), and pulling the n-gram vocabulary back from 150 to 100 features.
3. **Per-row Python object overhead.** `urls.apply(extract_features_row)` builds one Python dict per
   row before any DataFrame conversion — a dict's fixed per-instance overhead means 16.7M raw dicts
   need far more memory than the final compact float32 array would, entirely independent of fix #2.
   Fixed by processing in 500K-row chunks, converting each chunk to a compact array immediately.
4. **Unconstrained-depth Random Forests.** Both accounts survived static feature extraction (~7h)
   after fix #3, then crashed identically at the feature-importance-ranking RF fit — 200
   unconstrained-depth trees on the full 16.7M-row split grow enormous (this project's own production
   ensemble RF has trees with ~53-56k leaves each at similar scale). Fixed by subsampling the selector
   to 2M rows with `max_depth=20`, and pre-emptively applying `max_depth=30, max_samples=0.3` to the
   main production RF (which fits on an even larger ~18.8M-row post-SMOTE set) before it could cause
   a fifth crash.

Each fix was verified with a local smoke test (on a small row sample) before being re-pushed to
Kaggle, given how expensive a blind multi-hour retry cycle is at this scale.

## Data policy

`Dataset/` and `models/` are **never committed to git** (multi-gigabyte files don't belong in
version control) — `models/` artifacts are hosted on a private Hugging Face Hub repo and
auto-downloaded on demand by `model_registry.py` when missing locally. Secrets (`HF_TOKEN`,
`KAGGLE_API_TOKEN`, `GEMINI_API_KEY`, `ABUSE_CH_AUTH_KEY`) live in a local, gitignored `.env` file,
with cloud-deploy code paths checking real environment variables first and falling back to `.env`
for local development.

## Not yet started / explicitly out of scope

- A fusion layer correlating malware and URL risk scores into a single incident record.
- Wiring the completed ASN/partial WHOIS enrichment caches into the actual trained feature matrix
  (currently cache-only, used for live single-URL lookups and ablation studies, not bulk training —
  a deliberate, documented scope decision, not an oversight).
- Upgrading Module A (dynamic malware) and Module A'' (API-sequence) with the Avast-CTU CAPE dataset
  — identified as a strong option, deferred pending a new feature-extraction pipeline.
- SOREL-20M was confirmed public and reachable but its `pe_metadata` shard alone is 515GB+, far
  beyond what fits on this project's infrastructure — not downloaded. EMBER-2018 (already used in
  Module A') serves the same static-feature purpose at a fraction of the size.
