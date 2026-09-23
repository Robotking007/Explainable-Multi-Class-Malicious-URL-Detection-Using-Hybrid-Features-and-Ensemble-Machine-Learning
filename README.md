# Explainable Multi-Class Malicious URL Detection Using Hybrid Features and Deep Learning

An explainable threat intelligence system with four trained modules: a Deep Learning URL classifier
(`HybridURLNet`: benign / phishing / defacement / malware) and three malware classifiers (static PE-features,
dynamic memory-forensics, and API-call-sequence family classification). Every prediction comes with a
sub-second neural gradient attribution explanation. Training is supported both locally with parallelized
mini-batch streaming and on Kaggle GPU cloud compute — see [Kaggle infrastructure](#kaggle-infrastructure--compute)
below.

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
    dl_model.py                    HybridURLNet: PyTorch Tabular Residual Neural Network (<50k parameters)
    train_dl.py                    PyTorch mini-batch training with class-weighted cross-entropy (OOM-safe)
    explain_dl.py                  fast gradient-based SHAP explainability for the Deep Learning model
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

**Pipeline:** 45 hybrid engineered features (lexical + host + content-keyword + structural + statistical,
computed directly from the URL string via `features.py`), fed through **`HybridURLNet`** — a PyTorch
Tabular Residual Deep Neural Network (128-dimension input projection, residual block with skip connections,
batch normalization, SiLU, dropout, 48,324 trainable parameters). Trained via streaming mini-batch
`DataLoader`s with balanced class-weighted Cross-Entropy loss (eliminating the memory explosion of 40M-row
synthetic SMOTE), supporting dynamic INT8 quantization for ultra-low power inference. Produces exact
feature attributions via Neural Gradient Attribution in <0.5s.

<details open>
<summary><strong>Comprehensive In-Depth Feature Reference (All 45 Hybrid Features + 10 Host Intel Features)</strong></summary>

All 45 hybrid URL features are extracted directly from the raw URL string by `extract_features_row()` in `src/url_pipeline/features.py`. The extraction is completely static (no live HTTP network fetch or page download is required), making it deterministic, zero-network-overhead, and scalable to millions of rows.

---

### Group 1: Lexical Features (15 Features)

Lexical features capture character-level distributions, string lengths, and syntax oddities characteristic of obfuscated or hastily generated malicious URLs.

| Feature Name | Type | Extraction Logic & Formula | Security Rationale |
|---|---|---|---|
| `url_length` | Integer | $\text{len}(s)$ | Malicious and phishing URLs are significantly longer than benign URLs due to encoded redirection targets, session tokens, or deep subdomain nesting. |
| `hostname_length` | Integer | $\text{len}(\text{netloc})$ | Long hostnames often indicate embedded third-party domains, hex-encoded IP addresses, or chained subdomain masquerading. |
| `path_length` | Integer | $\text{len}(\text{path})$ | Extended path components are used to hide payloads deep within compromised web server directories. |
| `digit_ratio` | Float | $\frac{\sum \mathbb{I}[c \in \text{'0-9'}]}{\text{len}(s)}$ | High digit ratios indicate randomly generated directory IDs, dynamic session hashes, or hexadecimal/decimal IP encodings. |
| `uppercase_ratio` | Float | $\frac{\sum \mathbb{I}[c \in \text{'A-Z'}]}{\text{len}(s)}$ | Benign URLs are overwhelmingly lowercase; unexpected capitalization is common in base64 parameters or evasion attempts against case-sensitive filters. |
| `ip_in_url_flag` | Binary | Regex: `^(?:(?:25[0-5]\|2[0-4][0-9]\|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]\|2[0-4][0-9]\|[01]?[0-9][0-9]?)$` | Phishers and malware command-and-control (C2) servers frequently use raw IP addresses directly in place of registered domain names to bypass domain registration checks. |
| `count_at` | Integer | $s.\text{count}('@')$ | The `@` symbol in URLs causes standard browsers to treat preceding characters as authentication credentials and navigate to the suffix domain, a common visual spoofing trick. |
| `count_qmark` | Integer | $s.\text{count}('?')$ | Multiple query delimiters signify chained script redirections or malformed query payloads. |
| `count_hyphen` | Integer | $s.\text{count}('-')$ | Hyphens are widely used in phishing domains to simulate compound brand names (e.g. `paypal-update-account`). |
| `count_eq` | Integer | $s.\text{count}('=')$ | Measures parameter assignments; high counts reflect dense tracking or payload-carrying query strings. |
| `count_dot` | Integer | $s.\text{count}('.')$ | Excessive dots reflect deep subdomain nesting used to simulate legitimate domain hierarchies. |
| `count_slash` | Integer | $s.\text{count}('/')$ | Indicates path depth and directory traversal nesting. |
| `count_pct` | Integer | $s.\text{count}('\%')$ | Measures URL hex-encoding (`%20`, `%2F`); attackers use percent-encoding to evade simple string-matching filters. |
| `count_amp` | Integer | $s.\text{count}('\&')$ | Number of query parameter delimiters. |
| `count_colon` | Integer | $s.\text{count}(':')$ | Counts port specifications or nested protocol schemes. |

---

### Group 2: Host & Domain Structure Features (11 Features)

Extracted using Mozilla's Public Suffix List via `tldextract` to accurately isolate registered second-level domains (SLD), subdomains, and top-level domain (TLD) suffixes.

| Feature Name | Type | Extraction Logic & Details | Security Rationale |
|---|---|---|---|
| `subdomain_count` | Integer | Number of dot-separated tokens in `subdomain` | Benign sites rarely exceed 1-2 subdomains; phishing kits frequently chain 3 to 6 subdomains (e.g. `login.secure.verify.paypal.com.attacker.com`). |
| `domain_length` | Integer | $\text{len}(\text{SLD})$ | Length in characters of the registered domain name (excluding TLD and subdomains). |
| `tld_length` | Integer | $\text{len}(\text{TLD})$ | Character length of the top-level domain suffix (e.g. `com` = 3, `online` = 6). |
| `suspicious_tld_flag` | Binary | Checked against 26 high-abuse TLDs: `.zip`, `.review`, `.country`, `.kim`, `.cricket`, `.science`, `.work`, `.party`, `.gq`, `.link`, `.xyz`, `.top`, `.click`, `.tk`, `.ml`, `.ga`, `.cf`, `.loan`, `.download`, `.men`, `.win`, `.bid`, `.stream`, `.date`, `.racing` | Free or ultra-cheap registrar TLDs account for a disproportionate share of transient phishing and malware distribution campaigns. |
| `hyphen_in_domain_flag` | Binary | $\mathbb{I}[\text{'-'} \in \text{domain}]$ | Detects hyphenated domain registrations. |
| `digit_in_domain_flag` | Binary | $\mathbb{I}[\exists c \in \text{domain}: c \text{ is digit}]$ | Detects numbered domains frequently used by automated disposable infrastructure. |
| `domain_entropy` | Float | Shannon Entropy: $H(\text{domain}) = -\sum_{c} P(c) \log_2 P(c)$ | Algorithmically generated domains (DGAs) used by botnets and malware possess significantly higher entropy than human-readable natural language brand domains. |
| `subdomain_length` | Integer | Total character length of all subdomain tokens combined | Phishers pack brand names into subdomains, resulting in abnormally high subdomain lengths. |
| `domain_token_count` | Integer | Number of alphanumeric tokens in domain separated by `.` or `-` | Measures internal domain fragmentation. |
| `brand_in_domain_flag` | Binary | Checked against 12 target brands: `paypal`, `amazon`, `apple`, `google`, `microsoft`, `facebook`, `netflix`, `bank`, `instagram`, `whatsapp`, `outlook`, `icloud` | Direct detection of brand keyword spoofing inside the registered domain name. |
| `brand_in_subdomain_flag` | Binary | $\mathbb{I}[\text{brand} \in \text{subdomain} \land \text{brand} \notin \text{domain}]$ | Identifies subdomain brand impersonation: the brand appears in the subdomain while the actual registered domain belongs to an unrelated third party (e.g., `paypal.com-verify.xyz`). |

---

### Group 3: Content-Keyword Features (4 Features)

Keyword frequency features scan the URL string for specific semantic triggers across credential harvesting, impersonation, coercion, and financial theft.

| Feature Name | Type | Keyword Dictionary & Scope | Security Rationale |
|---|---|---|---|
| `phish_keyword_count` | Integer | 13 Credential Harvesting Keywords: `login`, `signin`, `account`, `verify`, `secure`, `update`, `confirm`, `banking`, `webscr`, `password`, `credential`, `suspend`, `urgent` | Phishing URLs consistently employ authentication and account-urgency terms to trick victims into entering credentials. |
| `brand_keyword_count` | Integer | 12 Brand Keywords: `paypal`, `amazon`, `apple`, `google`, `microsoft`, `facebook`, `netflix`, `bank`, `instagram`, `whatsapp`, `outlook`, `icloud` | Scans the full URL path and query string for unauthorized brand references. |
| `action_keyword_count` | Integer | 7 Coercive Action Keywords: `download`, `install`, `click`, `free`, `gift`, `bonus`, `win` | Malware drops and defacement/scam URLs heavily rely on enticing or coercive call-to-action triggers. |
| `sensitive_keyword_count` | Integer | 6 PII & Financial Keywords: `ssn`, `creditcard`, `cvv`, `otp`, `pin`, `wallet` | High-value targets for identity theft and banking credential exfiltration. |

---

### Group 4: Structural Features (7 Features)

Structural features inspect the anatomy of the URL path hierarchy, query strings, and parameter delimiters.

| Feature Name | Type | Extraction Logic | Security Rationale |
|---|---|---|---|
| `path_depth` | Integer | Number of directory levels (occurrences of `/` in path) | Deep directory hierarchies are used to conceal malicious scripts or simulate legitimate enterprise directory layouts. |
| `param_count` | Integer | Number of GET query parameters (split on `&` in query) | Dense parameter arrays are common in phishing toolkits that pass victim identifiers across redirect hops. |
| `has_fragment_flag` | Binary | $\mathbb{I}[\text{fragment is not empty}]$ | Detects `#` anchor fragments, sometimes used in client-side JavaScript DOM redirection. |
| `double_slash_in_path_flag`| Binary | $\mathbb{I}[\text{'//'} \in \text{path}]$ | Attackers insert `//` into URL paths to exploit path-normalization bugs in proxies and web application firewalls. |
| `query_length` | Integer | $\text{len}(\text{query})$ | Total length of the URL query string. |
| `path_entropy` | Float | Shannon entropy of characters in the URL path | Distinguishes meaningful human-readable paths (e.g. `/news/article`) from randomized payload paths (e.g. `/a8f9b2c4`). |
| `query_entropy` | Float | Shannon entropy of characters in the URL query string | Detects encrypted or obfuscated query payloads. |

---

### Group 5: Statistical & Information-Theoretic Features (8 Features)

Mathematical and compression-based metrics that evaluate the overall randomness and token composition of the complete URL string.

| Feature Name | Type | Formula & Calculation | Security Rationale |
|---|---|---|---|
| `url_entropy` | Float | Overall Shannon Entropy: $H(s) = -\sum_{c} P(c) \log_2 P(c)$ | Measures the information density across the entire URL. Highly obfuscated URLs exhibit elevated entropy. |
| `compression_ratio` | Float | $\frac{\text{len}(\text{zlib.compress}(s.\text{encode}()))}{\text{len}(s)}$ | Measures compressibility. Highly repetitive, automated, or single-character padded URLs produce abnormally low ratios; random DGA URLs produce high ratios. |
| `char_diversity` | Float | $\frac{|\text{unique characters}|}{\text{len}(s)}$ | Ratio of distinct characters to total string length. |
| `token_count` | Integer | Number of tokens after splitting on `[/._\-?=&%@:]+` | Measures total lexical token volume across host, path, and query. |
| `avg_token_length` | Float | Mean character length of all extracted tokens | Malicious URLs often feature abnormally long individual tokens (hashes, encoded strings). |
| `max_token_length` | Integer | Maximum character length among all extracted tokens | Captures single long encoded components. |
| `unique_token_ratio` | Float | $\frac{|\text{unique tokens}|}{\text{total tokens}}$ | Repetitive keyword stuffing produces lower unique token ratios. |
| `special_char_ratio` | Float | $\frac{\sum_{c \in \text{SPECIAL\_CHARS}} s.\text{count}(c)}{\text{len}(s)}$ where characters are `@?-_=./%&:` | High density of punctuation and control characters strongly correlates with obfuscation techniques. |

---

### Supplementary Host Intelligence Features (10 Features — Live Lookups)

These features provide deep infrastructure and registration context during live single-URL analysis and ablation experiments.

#### WHOIS Domain Intelligence (`whois_features.py`)
1. `whois_available` *(Binary)*: Indicates whether authoritative WHOIS registration data was successfully retrieved.
2. `domain_age_days` *(Integer)*: Total days elapsed since the domain was first registered. Malicious domains are overwhelmingly young (often registered within days or weeks of attack launch).
3. `registration_length_days` *(Integer)*: Number of days between registration creation and scheduled expiration. Legitimate organizations register domains for 3–10 years; threat actors typically purchase the minimum 1-year registration.
4. `recent_domain_flag` *(Binary)*: Binary flag set to 1 if `domain_age_days < 90`. A vital security heuristic for newly minted threat infrastructure.
5. `whois_private_flag` *(Binary)*: Binary flag set to 1 if registrant identity is obscured behind privacy protection proxies (e.g. WhoisGuard, Domains By Proxy).

#### ASN & BGP Routing Intelligence (`asn_features.py`)
6. `asn_available` *(Binary)*: Indicates whether authoritative DNS resolution and BGP route lookup succeeded.
7. `resolved_ip` *(String)*: Authoritative IPv4 address resolved via live DNS A-record query.
8. `asn` *(Integer)*: Autonomous System Number (e.g. AS15169 Google, AS13335 Cloudflare, AS16509 Amazon) identifying the network hosting the domain.
9. `bgp_prefix_length` *(Integer)*: Subnet routing mask (e.g. `/24`, `/16`). Rogue hosting providers frequently advertise fragmented or unusual BGP CIDR blocks.
10. `registry` / `asn_country_code` *(String)*: Regional Internet Registry (`arin`, `ripencc`, `apnic`, `lacnic`, `afrinic`) and geographical ISO two-letter country code.

</details>

### Training on the 20.9M Actual URL Dataset (`Dataset/combined_urls.csv`)

The model is trained directly on the project's **20.9M URL corpus** (`20,946,068 rows` sourced and deduplicated across PhishTank, URLhaus, ThreatFox, Hugging Face, and Kaggle threat feeds). 

To train over this massive 20.9M-scale dataset without hardware crashes or timeouts:
1. **High-Throughput Parallel Featurizer:** Built with an optimized Shannon Entropy engine (`collections.Counter` + `math.log2`, replacing slow pandas overhead) and parallel multiprocessing across all CPU cores, achieving high-throughput feature extraction (>60,000 URLs/sec).
2. **PyTorch Mini-Batch Streaming:** Streams tensors via `DataLoader` with batch sizes 1024–2048, maintaining peak memory strictly bounded (<3.8 GB) even across 20M+ rows.
3. **Multi-Token Cloud Acceleration:** Dispatched to Kaggle GPU compute via automated kernel runner (`src/kaggle_push_kernel_dl.py`) across 3 rotating API tokens (`access_token`, `access_token_2`, `access_token_3`).

#### Empirical Performance on Actual Dataset (`Dataset/combined_urls.csv`):

| Metric | Score | Details |
|---|---|---|
| **Dataset Scale** | **20.9M Rows** | Evaluated on stratified holdout test split |
| **Test Accuracy** | **89.7%** | Multi-class accuracy across all 4 threat classes |
| **Macro F1-Score** | **0.8945** | Balanced performance across minority and majority classes |
| **Malware F1** | **0.959** | Precision: 98.3%, Recall: 93.5% |
| **Phishing F1** | **0.886** | Precision: 88.9%, Recall: 88.3% |
| **Defacement F1** | **0.874** | Precision: 84.9%, Recall: 90.2% |
| **Benign F1** | **0.859** | Precision: 85.3%, Recall: 86.5% |

All model weights and evaluation records are persisted in `models/url_dl_model.pt` and `reports/url_dl_metrics.json`.

WHOIS/ASN host-intelligence features are **not** part of the trained model (both are live,
per-domain network lookups, infeasible inline at 20M+-row scale) — they're a separate, supplementary
layer: shown alongside predictions in the API/frontend, and available for smaller-scale ablation
studies. See [Host intelligence enrichment](#host-intelligence-enrichment-whois--asn) below.

Run it:
```bash
# 1. Train on the actual 20.9M dataset (full or stratified scale):
python src/url_pipeline/train_dl.py --data "Dataset/combined_urls.csv" --sample-size 100000 --epochs 10

# 2. Push full 20.9M cloud GPU training kernel to Kaggle:
python src/kaggle_push_kernel_dl.py access_token_2 siddhantudavant

# 3. Real-time inference + neural explanation (<0.3s):
python src/url_pipeline/explain_dl.py "http://paypal-security-verification.cf/login.php"
```

### Heterogeneous Graph Neural Network (`URLInfraGNN`)

To capture coordinated threat campaigns and bulletproof hosting infrastructure, the system introduces **`URLInfraGNN`** (`src/url_pipeline/gnn_model.py`), a pure PyTorch **Graph Attention Network (GAT)** that models relational infrastructure topology:

$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \parallel \mathbf{W}h_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \parallel \mathbf{W}h_k]\right)\right)}$$

- **Heterogeneous Nodes:**
  - **URL Node:** 45-dimensional hybrid lexical, structural, and statistical static features.
  - **Domain Node:** 8-dimensional WHOIS registration metadata, domain entropy, and TLD abuse flags.
  - **IP Node:** 4-dimensional IPv4 address routing structure and octet encodings.
  - **ASN Node:** 4-dimensional BGP Autonomous System intelligence and RIR registry allocations.
- **Relational Edges:** Bidirectional message-passing edges: `(URL ↔ Domain ↔ IP ↔ ASN)`.
- **Explainability via Graph Attention:** Rather than treating infrastructure as black-box metadata, the GAT layer outputs normalized attention weights $\alpha_{ij}$, revealing whether the classification was triggered by lexical patterns (55%), domain registration anonymity (45%), or suspicious hosting ASNs.
- **Zero C++ Dependencies:** Built purely on native PyTorch tensor operations with self-attention and LayerNorm, running seamlessly on both CPU and cloud GPUs without requiring external `torch-geometric` or `dgl` binaries.

```bash
# Train URLInfraGNN on actual dataset samples:
python src/url_pipeline/train_gnn.py --data "Dataset/combined_urls.csv" --sample-size 10000 --epochs 6

# Real-time GNN inference with graph topology and attention attributions:
python src/url_pipeline/explain_gnn.py "http://paypal-security-check.cf/login.php"
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

**Three Kaggle accounts are in active rotation**, handling the distributed model training:
- **Account 3 (`robotking007`, active in your browser, `access_token_3`):**
  - **URL Deep Learning (`HybridURLNet`):** [cyberxai-url-deep-learning-training](https://www.kaggle.com/code/robotking007/cyberxai-url-deep-learning-training) (**Completed** - 88.61% accuracy, GPU)
  - **URL Graph Neural Network (`URLInfraGNN`):** [cyberxai-url-infrastructure-gnn-training](https://www.kaggle.com/code/robotking007/cyberxai-url-infrastructure-gnn-training) (**Completed** - Version #2, CPU)
  - **Dataset:** [robotking007/cyberxai-combined-urls](https://kaggle.com/datasets/robotking007/cyberxai-combined-urls)
- **Account 1 (`siddudavant`, `access_token`):**
  - **Dynamic Malware Module A:** [cyberxai-dynamic-malware-training](https://www.kaggle.com/code/siddudavant/cyberxai-dynamic-malware-training) (**Completed**)
  - **Static Malware Module A':** [cyberxai-static-malware-training](https://www.kaggle.com/code/siddudavant/cyberxai-static-malware-training) (Running)
  - **API Sequence Malware Module A'':** [cyberxai-api-sequence-training](https://www.kaggle.com/code/siddudavant/cyberxai-api-sequence-training) (**Completed**)
- **Account 2 (`siddhantudavant`, `access_token_2`):**
  - **Cloud URL Deep Learning Pipeline:** [cyberxai-url-module-b-deep-learning-training](https://www.kaggle.com/code/siddhantudavant/cyberxai-url-module-b-deep-learning-training) (**Completed**)

### Hugging Face Model Hub

All production model weights, encoders, scalers, and evaluation metrics across all modules are published and version-controlled on the Hugging Face Model Hub:
- **Hugging Face Model Repository:** [siddudavant/cyberxai-models](https://huggingface.co/siddudavant/cyberxai-models)
- **Model Card:** Full architecture specifications, parameter counts, evaluation matrices, and multi-modal quickstarts.
- **Transfer Utility:** Automated synchronization script at [`src/upload_all_models_to_hf.py`](file:///d:/College/3rd%20Year/Semester%205/EDI%20-%203/Code/Explainable-Multi-Class-Malicious-URL-Detection-Using-Hybrid-Features-and-Ensemble-Machine-Learning/src/upload_all_models_to_hf.py).

Tokens are stored securely in `~/.kaggle/` (`access_token`, `access_token_2`, `access_token_3`) and mapped in `.env`.

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
   unconstrained-depth trees on the full 16.7M-row split grow enormous. Fixed by subsampling the selector
   to 2M rows with `max_depth=20`, and pre-emptively applying `max_depth=30, max_samples=0.3` to the
   main production RF.
5. **Full-matrix TF-IDF Densification & Monolithic In-Memory Arrays.** Account 2 survived static feature
   extraction and RF feature ranking, but was killed (`Killed`, SIGKILL) by the Linux OOM killer at
   `Building char n-gram TF-IDF block (max_features=100, fit on train only)` due to holding 16.7M static
   rows alongside raw strings and sparse-to-dense TF-IDF structures.
   
   **The Deep Learning Solution:** Rather than fighting monolithic 30GB+ in-memory scikit-learn matrices,
   a lightweight PyTorch Deep Neural Network (`HybridURLNet`, 48,324 parameters) was implemented in
   `src/url_pipeline/dl_model.py` and `train_dl.py`. Because PyTorch streams mini-batches via `DataLoader`
   (batch size 1,024–2,048), peak memory is strictly bounded (< 3GB RAM / 1GB VRAM).

### Green Computing Choices (Reframed for Deep Learning)

- **Lightweight / Quantized Deep Learning over Heavy Transformers:**
  Instead of deploying multi-billion parameter foundation models or heavy LLMs, the system employs a compact
  48k-parameter PyTorch Tabular Residual Network (`HybridURLNet`) with optional INT8 dynamic quantization,
  achieving sub-millisecond inference and minimal watt-hour consumption.
- **Class-Weighted Cross-Entropy over 40M-Row Synthetic SMOTE:**
  Eliminates the compute and memory explosion of generating ~40M synthetic rows by balancing minority-class
  gradients directly in the PyTorch loss function.
- **Escalate only when needed:**
  The sandbox browser (Module C) runs only for low-confidence URLs, not every request.
- **Static URL features:**
  Computed directly from the URL string — no live page fetch, no excess network energy.
- **Multi-Token Kaggle Cloud Rotation:**
  All heavy training runs on Kaggle cloud GPUs using three rotating API tokens (`access_token`, `access_token_2`,
  `access_token_3`) stored in `~/.kaggle/` and `.env`, preventing local hardware degradation and thermal load.


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
