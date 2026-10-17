"""
Self-contained Module B (URL) training script for a Kaggle kernel.

This is a standalone copy of src/url_pipeline/train.py + the feature
extraction it depends on (src/url_pipeline/features.py), inlined into one
file because a Kaggle kernel can't import from this repo's own package
layout. Logic is otherwise identical to train.py, including the two
memory-crash fixes (n_jobs=1 for the unconstrained-depth RFs, memmap for
the SMOTE-balanced matrix) -- kept in even though Kaggle's ~30GB RAM gives
much more headroom than the 16GB local machine that needed them, as cheap
insurance against the same failure mode recurring.

Dataset expected at /kaggle/input/cyberxai-combined-urls/combined_urls.csv
(uploaded separately via src/kaggle_upload_dataset.py). Outputs saved to
/kaggle/working/ for download via the Kernels Output API afterward.
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn", "tldextract"], check=False)

import gc
import json
import math
import re
import time
import warnings
import zlib

warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import tldextract
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

DATA_PATH = Path("/kaggle/input/cyberxai-combined-urls/combined_urls.csv")
OUT_DIR = Path("/kaggle/working")
TOP_K = 30
NGRAM_FEATURES = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Cap for the two smallest classes (defacement/malware, ~200k/~187k) after
# SMOTE -- NOT full parity with benign (10.89M). At 20M+ total rows,
# fully balancing every class to match benign would need ~40M+ rows,
# mostly synthetic/interpolated, which is both computationally
# impractical and actively hurts generalization (SMOTE's linear
# interpolation over sparse n-gram features tends to manufacture
# unrealistic points rather than real diversity at this ratio). A moderate
# 1.2M-row cap gives real headroom over the raw counts without drowning
# the signal in synthetic noise, and class_weight="balanced" (RF/LR) plus
# sample_weight (XGBoost) handles the remaining imbalance via the loss
# function instead of via manufactured data.
SMOTE_TARGET_PER_MINORITY_CLASS = 1_200_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================== features.py (inlined) ==============================
SUSPICIOUS_TLDS = {
    "zip", "review", "country", "kim", "cricket", "science", "work",
    "party", "gq", "link", "xyz", "top", "click", "tk", "ml", "ga", "cf",
    "loan", "download", "men", "win", "bid", "stream", "date", "racing",
}
PHISH_KEYWORDS = [
    "login", "signin", "account", "verify", "secure", "update", "confirm",
    "banking", "webscr", "password", "credential", "suspend", "urgent",
]
BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "facebook",
    "netflix", "bank", "instagram", "whatsapp", "outlook", "icloud",
]
ACTION_KEYWORDS = ["download", "install", "click", "free", "gift", "bonus", "win"]
SENSITIVE_KEYWORDS = ["ssn", "creditcard", "cvv", "otp", "pin", "wallet"]
IP_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)
SPECIAL_CHARS = ["@", "?", "-", "=", ".", "/", "%", "&", ":"]
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = pd.Series(list(s)).value_counts()
    probs = counts / len(s)
    return float(-(probs * np.log2(probs)).sum())


def normalize_url(raw_url: str) -> str:
    # Strips a literal wrapping single-quote (CSV-quoting artifact from
    # MUD_malicious_urls_2026_V2.csv/merged_urls_dataset.csv) -- see
    # features.py's normalize_url docstring for the full story. Scheme
    # must be stripped BEFORE the quote check: merged_urls_dataset.csv's
    # variant is "http://'domain'" (quote after the scheme), so checking
    # for a leading quote first silently misses every row from that source.
    s = str(raw_url).strip()
    s = _SCHEME_RE.sub("", s)
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1]
        s = _SCHEME_RE.sub("", s)
    return s


def _safe_urlparse(url: str):
    u = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", u):
        u = "http://" + u
    try:
        return urlparse(u)
    except Exception:
        return urlparse("http://invalid")


def extract_features_row(raw_url: str) -> dict:
    url = normalize_url(raw_url)
    parsed = _safe_urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    ext = tldextract.extract(url)
    domain = ext.domain or ""
    subdomain = ext.subdomain or ""
    suffix = ext.suffix or ""
    # top_domain_under_public_suffix, not the deprecated registered_domain --
    # same value, but doesn't emit a DeprecationWarning on every single call.
    # At ~20M rows that warning (even though Python's default filter should
    # dedupe by call site) was somehow repeating enough to completely fill
    # Kaggle's returned log buffer, hiding every real log() line after
    # static-feature extraction started -- exactly the output needed to
    # tell where the OOM kill actually happened.
    registered_domain = ext.top_domain_under_public_suffix or hostname
    lower_url = url.lower()
    feats = {}

    feats["url_length"] = len(url)
    feats["hostname_length"] = len(hostname)
    feats["path_length"] = len(path)
    digits = sum(c.isdigit() for c in url)
    uppers = sum(c.isupper() for c in url)
    feats["digit_ratio"] = digits / max(len(url), 1)
    feats["uppercase_ratio"] = uppers / max(len(url), 1)
    feats["ip_in_url_flag"] = int(bool(IP_RE.match(hostname)))
    for ch in SPECIAL_CHARS:
        feats[f"count_{ch if ch.isalnum() else {'@':'at','?':'qmark','-':'hyphen','=':'eq','.':'dot','/':'slash','%':'pct','&':'amp',':':'colon'}[ch]}"] = url.count(ch)

    feats["subdomain_count"] = subdomain.count(".") + (1 if subdomain else 0)
    feats["domain_length"] = len(domain)
    feats["tld_length"] = len(suffix)
    feats["suspicious_tld_flag"] = int(suffix.split(".")[-1] in SUSPICIOUS_TLDS if suffix else False)
    feats["hyphen_in_domain_flag"] = int("-" in domain)
    feats["digit_in_domain_flag"] = int(any(c.isdigit() for c in domain))
    feats["domain_entropy"] = _shannon_entropy(registered_domain)
    feats["brand_in_subdomain_flag"] = int(
        any(b in subdomain.lower() for b in BRAND_KEYWORDS) and domain.lower() not in BRAND_KEYWORDS
    )
    feats["subdomain_length"] = len(subdomain)
    feats["domain_token_count"] = len([t for t in re.split(r"[-.]", domain) if t])
    feats["brand_in_domain_flag"] = int(any(b in domain.lower() and domain.lower() != b for b in BRAND_KEYWORDS))

    feats["phish_keyword_count"] = sum(k in lower_url for k in PHISH_KEYWORDS)
    feats["brand_keyword_count"] = sum(k in lower_url for k in BRAND_KEYWORDS)
    feats["action_keyword_count"] = sum(k in lower_url for k in ACTION_KEYWORDS)
    feats["sensitive_keyword_count"] = sum(k in lower_url for k in SENSITIVE_KEYWORDS)

    feats["path_depth"] = path.count("/")
    feats["param_count"] = query.count("&") + (1 if query else 0)
    feats["has_fragment_flag"] = int(bool(fragment))
    feats["double_slash_in_path_flag"] = int("//" in path)
    feats["query_length"] = len(query)
    feats["path_entropy"] = _shannon_entropy(path)
    feats["query_entropy"] = _shannon_entropy(query)

    feats["url_entropy"] = _shannon_entropy(url)
    compressed_len = len(zlib.compress(url.encode("utf-8", errors="ignore")))
    feats["compression_ratio"] = compressed_len / max(len(url), 1)
    feats["char_diversity"] = len(set(url)) / max(len(url), 1)
    tokens = [t for t in re.split(r"[/\-_.?=&%@:]+", url) if t]
    token_lengths = [len(t) for t in tokens]
    feats["token_count"] = len(tokens)
    feats["avg_token_length"] = float(np.mean(token_lengths)) if token_lengths else 0.0
    feats["max_token_length"] = max(token_lengths) if token_lengths else 0
    feats["unique_token_ratio"] = (len(set(tokens)) / len(tokens)) if tokens else 0.0
    special_chars = sum(1 for c in url if not c.isalnum())
    feats["special_char_ratio"] = special_chars / max(len(url), 1)
    return feats


def build_static_features(urls: pd.Series, chunk_size: int = 500_000) -> pd.DataFrame:
    # OOM history (3rd crash, acct2 run): the previous two fixes (float32
    # casts, sequential build+free) targeted the FINAL columnar arrays, but
    # this run died even earlier, right at the start of static feature
    # extraction itself -- before any DataFrame/float32 conversion even
    # happens. Root cause: `urls.apply(extract_features_row)` builds one
    # Python dict per row (45 keys each) and holds ALL of them in memory
    # as a pandas Series of objects before pd.DataFrame() ever converts
    # them to compact columnar arrays. A dict has substantial fixed
    # overhead per instance (hash table, key/value slots) -- at 16.7M
    # rows this raw-object intermediate form alone plausibly needs tens
    # of GB, regardless of how efficiently the FINAL float32 DataFrame
    # would have stored the same data.
    #
    # Fix: process in chunks, converting each chunk's dicts to a compact
    # float32 DataFrame and discarding the raw dicts immediately, so at
    # most chunk_size dicts (not all 16.7M) are ever alive at once.
    chunks = []
    n = len(urls)
    for start in range(0, n, chunk_size):
        chunk_urls = urls.iloc[start:start + chunk_size].astype(str)
        records = chunk_urls.apply(extract_features_row)
        chunk_df = pd.DataFrame(list(records))
        float_cols = chunk_df.select_dtypes(include="float64").columns
        chunk_df[float_cols] = chunk_df[float_cols].astype(np.float32)
        int_cols = chunk_df.select_dtypes(include="int64").columns
        chunk_df[int_cols] = chunk_df[int_cols].astype(np.int32)
        chunks.append(chunk_df)
        del records, chunk_df, chunk_urls
        gc.collect()
    return pd.concat(chunks, ignore_index=True)


def build_ngram_features(train_urls: pd.Series, test_urls: pd.Series, max_features: int = 100):
    # OOM history: at ~20M rows, TfidfVectorizer's default float64 sparse
    # output, densified via .toarray(), needs rows*max_features*8 bytes --
    # at max_features=150 that's ~19GB for the train split alone (16M rows),
    # which OOM-killed a full training run after 4.3h of otherwise-successful
    # static feature extraction (no crash traceback -- the Linux OOM killer
    # just SIGKILLs the process, nothing to catch). Casting the sparse
    # matrix to float32 BEFORE densifying (not after) means the dense array
    # is float32 from the moment it's allocated -- never briefly exists as
    # the larger float64 form -- roughly halving this block's peak memory.
    from sklearn.feature_extraction.text import TfidfVectorizer
    train_norm = train_urls.astype(str).apply(normalize_url)
    test_norm = test_urls.astype(str).apply(normalize_url)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=max_features)
    X_train = vectorizer.fit_transform(train_norm).astype(np.float32)
    X_test = vectorizer.transform(test_norm).astype(np.float32)
    cols = [f"ngram_{i}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train.toarray(), columns=cols)
    test_df = pd.DataFrame(X_test.toarray(), columns=cols)
    return train_df, test_df, vectorizer


def fit_ngram_vectorizer(train_urls: pd.Series, max_features: int = 100):
    """Split out of build_ngram_features so main() can fit+densify the train
    n-gram block, use it, and free it BEFORE building the test block --
    the two were previously built together, so both stayed alive at once
    even though the train block is only needed early (for the final
    concat) and the test block only later (for evaluation). Sequencing
    them cuts this block's peak memory roughly in half again."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    train_norm = train_urls.astype(str).apply(normalize_url)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=max_features)
    X_train = vectorizer.fit_transform(train_norm).astype(np.float32)
    cols = [f"ngram_{i}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train.toarray(), columns=cols)
    return train_df, vectorizer


def transform_ngram_features(vectorizer, urls: pd.Series) -> pd.DataFrame:
    norm = urls.astype(str).apply(normalize_url)
    X = vectorizer.transform(norm).astype(np.float32)
    cols = [f"ngram_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X.toarray(), columns=cols)


def detect_xgb_device() -> str:
    try:
        import xgboost as xgb_mod
        d = xgb_mod.DMatrix(np.zeros((4, 2)), label=[0, 1, 0, 1])
        xgb_mod.train({"tree_method": "hist", "device": "cuda"}, d, num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


def compute_fnr_fpr(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fnr, fpr = {}, {}
    total = cm.sum()
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        fnr[cls] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr[cls] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fnr, fpr


# ============================== main ==============================
def main():
    log("Listing /kaggle/input/ to confirm actual dataset mount path...")
    for p in Path("/kaggle/input").rglob("*"):
        log(f"  {p}")

    data_path = DATA_PATH
    if not data_path.exists():
        candidates = list(Path("/kaggle/input").rglob("combined_urls.csv"))
        if candidates:
            data_path = candidates[0]
            log(f"DATA_PATH did not exist; found actual file at {data_path} instead")
        else:
            raise FileNotFoundError(f"combined_urls.csv not found anywhere under /kaggle/input (expected {DATA_PATH})")

    log(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path).dropna(subset=["url", "type"])
    log(f"Class distribution:\n{df['type'].value_counts()}")

    log("Splitting train/test (stratified, 80/20)")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["type"])

    y_train = train_df["type"].reset_index(drop=True)
    y_test = test_df["type"].reset_index(drop=True)
    train_urls = train_df["url"].reset_index(drop=True)
    test_urls = test_df["url"].reset_index(drop=True)
    del df, train_df, test_df
    gc.collect()

    # OOM history: two prior runs on this ~20.9M-row dataset were SIGKILLed
    # by the Linux OOM killer (no traceback -- OOM kills leave nothing to
    # catch). float32 dtypes alone (a first fix attempt) weren't enough --
    # the real problem was that the OLD code built X_train_static (45 cols),
    # X_train_ngram (100 cols), X_train_full (=concat of both, a NEW copy),
    # and X_train_sel (=a subset of X_train_full, ANOTHER new copy) for
    # both train AND test, all without freeing any of the precursors until
    # everything was built -- up to ~7 large, nearly-redundant copies of
    # the same rows alive simultaneously (estimated ~35-40GB peak on the
    # train split alone). This version processes train and test
    # sequentially and deletes each precursor immediately after its last
    # use, so at most ~2 large arrays are ever alive at once.
    log(f"Extracting static features on {len(train_urls)} train + {len(test_urls)} test URLs ...")
    t0 = time.time()
    X_train_static = build_static_features(train_urls)
    log(f"Train static features done in {time.time() - t0:.1f}s. Shape: {X_train_static.shape}")

    # OOM history (4th crash, both accounts): the previous three fixes got
    # past dataset load, train/test split, AND static feature extraction
    # (confirmed by the log reaching "Train static features done" both
    # times) -- but died immediately on rf_selector.fit(), an UNCONSTRAINED-
    # DEPTH RandomForest with 200 trees fit on the full 16.7M-row training
    # split. Unconstrained trees at that scale grow extremely large (this
    # project's own docs elsewhere note ~53-56k leaves/tree for a similarly
    # unconstrained 300-tree RF), and RandomForestClassifier keeps every
    # fitted tree in memory afterward, not just one at a time. This
    # selector only needs a feature-importance RANKING (which 45 of the
    # engineered features matter most), not a production-quality model --
    # a shallow forest on a modest random subsample gives an equally
    # reliable ranking at a small fraction of the memory/time cost.
    log("Ranking engineered features by Random Forest importance (on a subsample, shallow trees) ...")
    rng = np.random.RandomState(RANDOM_STATE)
    sample_size = min(2_000_000, len(X_train_static))
    sample_idx = rng.choice(len(X_train_static), size=sample_size, replace=False)
    X_selector_sample = X_train_static.iloc[sample_idx]
    y_selector_sample = y_train.iloc[sample_idx]
    rf_selector = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=1
    )
    rf_selector.fit(X_selector_sample, y_selector_sample)
    del X_selector_sample, y_selector_sample
    gc.collect()
    importances = pd.Series(rf_selector.feature_importances_, index=X_train_static.columns)
    top_k_cols = importances.sort_values(ascending=False).head(TOP_K).index.tolist()
    log(f"Top-{TOP_K} engineered features selected:\n{importances.sort_values(ascending=False).head(TOP_K)}")
    del rf_selector
    X_train_static_sel = X_train_static[top_k_cols].copy()
    del X_train_static
    gc.collect()

    t0 = time.time()
    X_test_static = build_static_features(test_urls)
    X_test_static_sel = X_test_static[top_k_cols].copy()
    del X_test_static
    gc.collect()
    log(f"Test static features done in {time.time() - t0:.1f}s.")

    log(f"Building char n-gram TF-IDF block (max_features={NGRAM_FEATURES}, fit on train only)")
    X_train_ngram, vectorizer = fit_ngram_vectorizer(train_urls, max_features=NGRAM_FEATURES)
    ngram_cols = list(X_train_ngram.columns)
    selected_cols = top_k_cols + ngram_cols

    X_train_sel = pd.concat(
        [X_train_static_sel.reset_index(drop=True), X_train_ngram.reset_index(drop=True)], axis=1
    )[selected_cols]
    del X_train_static_sel, X_train_ngram
    gc.collect()

    X_test_ngram = transform_ngram_features(vectorizer, test_urls)
    X_test_sel = pd.concat(
        [X_test_static_sel.reset_index(drop=True), X_test_ngram.reset_index(drop=True)], axis=1
    )[selected_cols]
    del X_test_static_sel, X_test_ngram
    gc.collect()
    log(f"Combined + selected feature matrix: {len(selected_cols)} features ({len(top_k_cols)} engineered + {len(ngram_cols)} n-gram)")

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=selected_cols, dtype=np.float32)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_sel), columns=selected_cols, dtype=np.float32)
    del X_train_sel, X_test_sel
    gc.collect()

    log(f"Class counts before SMOTE:\n{y_train.value_counts()}")
    # Capped, targeted SMOTE: only oversample classes below the cap, and
    # only up to the cap -- NOT full parity with benign. See
    # SMOTE_TARGET_PER_MINORITY_CLASS's definition above for why full
    # parity at 20M+ rows would be both impractical and counterproductive.
    counts = y_train.value_counts()
    sampling_strategy = {
        cls: SMOTE_TARGET_PER_MINORITY_CLASS
        for cls, n in counts.items()
        if n < SMOTE_TARGET_PER_MINORITY_CLASS
    }
    log(f"SMOTE sampling_strategy (capped, targeted): {sampling_strategy}")
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    X_train_bal_np = X_train_bal.astype(np.float32).to_numpy()
    del X_train_bal, X_train_scaled
    gc.collect()
    log(f"Class counts after capped SMOTE:\n{pd.Series(y_train_bal).value_counts()}")

    tmp_dir = OUT_DIR / "_tmp_arrays"
    tmp_dir.mkdir(exist_ok=True)
    mm_path = tmp_dir / "X_train_bal_url.f32"
    X_train_bal = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=X_train_bal_np.shape)
    X_train_bal[:] = X_train_bal_np
    X_train_bal.flush()
    del X_train_bal_np
    gc.collect()

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_bal)
    y_test_enc = le.transform(y_test)
    classes = list(le.classes_)

    # Remaining imbalance (benign ~10.9M vs. even the SMOTE-capped 1.2M
    # defacement/malware) is handled via sample_weight rather than more
    # synthetic oversampling -- VotingClassifier.fit(sample_weight=...)
    # forwards the same weights to every member that supports them (RF,
    # XGBoost, LR, GaussianNB all do), so this one array balances the loss
    # for all four voters without needing per-estimator class_weight.
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weight = compute_sample_weight("balanced", y_train_enc)
    log(f"Computed balanced sample_weight over {len(np.unique(y_train_enc))} classes.")

    xgb_device = detect_xgb_device()
    log(f"Training Voting Classifier (RF + XGBoost[{xgb_device}] + LR + NB) ...")
    # max_depth/max_samples added after the rf_selector crash above proved
    # unconstrained-depth RF trees are what's OOM-killing this pipeline at
    # ~17-19M-row scale, not (only) the surrounding data-copy overhead
    # fixed earlier. This RF fits on the SMOTE-balanced set (~18.8M rows:
    # raw benign+phishing, capped defacement/malware) -- even larger than
    # the 16.7M-row rf_selector fit that just crashed both accounts after
    # ~7h each. max_depth=30 and max_samples=0.3 (each tree bootstraps
    # ~30% of rows) bound both the per-tree node count and the per-tree
    # training-sample size, which is what unconstrained depth blows up
    # uncontrollably at this scale. 300 trees kept (vs. fewer/shallower)
    # to preserve as much ensemble diversity/accuracy as this constraint
    # budget allows -- tree COUNT costs much less memory than tree DEPTH.
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=30, max_samples=0.3,
        random_state=RANDOM_STATE, n_jobs=1,
    )
    xgb = XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.08, random_state=RANDOM_STATE,
        n_jobs=-1, eval_metric="mlogloss", tree_method="hist", device=xgb_device,
    )
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    nb = GaussianNB()

    voting = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb), ("lr", lr), ("nb", nb)], voting="soft", n_jobs=None)
    t0 = time.time()
    voting.fit(X_train_bal, y_train_enc, sample_weight=sample_weight)
    log(f"Voting Classifier trained in {time.time() - t0:.1f}s")

    log("Evaluating on held-out test split ...")
    y_pred_enc = voting.predict(X_test_scaled)
    y_proba = voting.predict_proba(X_test_scaled)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        roc_auc_macro = roc_auc_score(y_test_enc, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc_macro = float("nan")
    fnr, fpr = compute_fnr_fpr(y_test, y_pred, classes)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    log(f"Accuracy: {acc:.4f} | Macro-Precision: {prec_macro:.4f} | Macro-Recall: {rec_macro:.4f} | Macro-F1: {f1_macro:.4f} | Macro-ROC-AUC: {roc_auc_macro:.4f}")
    log(f"Per-class classification report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    joblib.dump(voting, OUT_DIR / "url_voting_classifier.joblib")
    joblib.dump(scaler, OUT_DIR / "url_scaler.joblib")
    joblib.dump(le, OUT_DIR / "url_label_encoder.joblib")
    joblib.dump(vectorizer, OUT_DIR / "url_ngram_vectorizer.joblib")
    with open(OUT_DIR / "url_selected_features.json", "w") as f:
        json.dump(selected_cols, f, indent=2)

    metrics_out = {
        "n_train": len(X_train_bal), "n_test": len(X_test_scaled), "n_features": len(selected_cols),
        "top_k_engineered_features": top_k_cols, "accuracy": acc, "macro_precision": prec_macro,
        "macro_recall": rec_macro, "macro_f1": f1_macro, "macro_roc_auc": roc_auc_macro,
        "per_class_fnr": fnr, "per_class_fpr": fpr, "classification_report": report_dict,
    }
    with open(OUT_DIR / "url_module_b_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    log(f"Saved model artifacts and metrics to {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "CRASH_TRACEBACK.txt", "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        raise
