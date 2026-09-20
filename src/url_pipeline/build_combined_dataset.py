"""
Merge every usable URL dataset in Dataset/ into a single, deduplicated
training corpus for Module B.

Sources are merged in this explicit priority order (highest first) -- for
the small fraction of URLs that appear in more than one source with a
DIFFERENT label, the highest-priority source's label wins:

  1. Dataset/verified_online.csv        - PhishTank's live-verified feed
     (phishing-only). Most authoritative for "is this phishing": each entry
     has been human-verified, not just scraped.
  2. Dataset/file.csv.txt               - abuse.ch URLhaus dump (malware-only,
     downloaded locally; NOT a live API call -- see whois_features.py's
     docstring for the project's policy on the abuse.ch key, which this does
     not use). Same authority tier as verified_online.csv, but for the
     malware class: a live, curated malware-distribution threat feed, not a
     scraped/unverified corpus.
  3. Dataset/MUD_malicious_urls_2026_V2.csv - newer (2026), larger corpus
     with its own feature/label pipeline.
  4. Dataset/malicious_phish.csv        - original corpus (2021/2022 vintage).
  5. Dataset/merged_urls_dataset.csv    - unknown provenance, 3.9M rows total
     but 3.37M are labeled "adult" (out of scope for this project's 4-class
     taxonomy and dropped entirely); the remaining benign/phish/malware rows
     are kept.
  6. Dataset/PhishVN.../data/dataset_url.csv - PhishVN v3.1.0 (Vietnamese
     URL phishing dataset), extracted from its own zip. Well-documented
     research dataset (datasheet/schema/citation included) with clean
     phishing/benign labels, verified by sampling: phishing entries are
     plausible typosquats, benign entries are real .gov.vn domains.
  7. Everything in ADDITIONAL_SOURCES below (see its own comments) -- newer
     datasets added to Dataset/ after the first four/five were already
     established as the authoritative core. Binary (benign/malicious-only)
     sources whose "malicious" side doesn't specify WHICH of
     phishing/defacement/malware it is are only used for their benign rows
     (see SKIP_AMBIGUOUS_MALICIOUS below) -- guessing a specific class for
     an ambiguously-labeled row would poison that class rather than grow it.

A second candidate source (Dataset/Phishing Website Detection Dataset/,
6.5M headerless plaintext URLs) was investigated and deliberately EXCLUDED:
sampling its "malign" file found 14,641 lines referencing amazon.com/
google.com/wikipedia.org/microsoft.com, plus many adult-content blogspot
URLs in both classes -- strong evidence it's an adult-content classifier
mislabeled as a phishing dataset, not a phishing/malware corpus at all.
Merging it would have taught the model that mainstream sites and Microsoft's
own mail-protection domain are malicious -- the same class of labeling bug
as the scheme-prefix leak this project already found and fixed once.

Label polarity for binary (0/1) sources was verified by inspecting actual
sample URLs per label value, not assumed -- two sources
(url_features_extracted1.csv, PhiUSIIL_Phishing_URL_Dataset.csv) turned out
to use the OPPOSITE convention (1=legitimate/benign, 0=phishing) from the
other binary sources (Dataset.csv, DomainGuard-50K.csv, dataset2.csv), which
use 0=benign, 1=phishing. Getting this backwards would have silently
mislabeled hundreds of thousands of rows, the same class of bug as the
scheme-prefix leak this project already found and fixed once.

URLs are normalized the same way features.py does (scheme stripped) before
deduplication, so "http://x.com" and "https://x.com" collapse to one row
rather than being counted twice or, worse, appearing in different classes.

Output: Dataset/combined_urls.csv (columns: url, type)

Usage:
    python src/url_pipeline/build_combined_dataset.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from url_pipeline.features import normalize_url  # noqa: E402

DATASET_DIR = ROOT / "Dataset"
VALID_CLASSES = {"benign", "phishing", "defacement", "malware"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Each entry: (filename, url_col, label_col, mapping).
# `mapping` is either a dict {raw_label_value: our_class}, or the string
# constant to assign to every row (for single-class sources like the
# phishing-only / malware-only feeds). A raw label value not present in
# `mapping` is dropped (e.g. "spam" -- not one of our 4 classes).
ADDITIONAL_SOURCES = [
    # 12M-row Kaggle dataset (zhengxian142857/phishing-legitimate-website-
    # link-and-domain), clean url/label schema, ~balanced phishing/legitimate.
    ("zhengxian_12M_urls.csv", "input", "label", {"legitimate": "benign", "phishing": "phishing"}),
    # 6.1M-row Kaggle dataset (somanshumahajan/65-lakh-urls-labelled-as-1-and-0),
    # verified 0=benign/1=phishing by URL sampling (label=0 rows are real
    # sites like mclarenstore.com; label=1 rows are clear typosquats like
    # "blizzardenteaitaccoutmanagementv.com" and "chase.com.asp72.com").
    ("somanshumahajan_6M_urls.csv", "url", "label", {0: "benign", 1: "phishing"}),
    # HF dataset kmack/Phishing_urls (708,820 rows, train+test+valid
    # concatenated), verified 0=benign/1=phishing via a full-column sample
    # (label=0: en.wikipedia.org, fi.edu, unsw.edu.au; label=1: brand-
    # impersonation subdomains, typosquats).
    ("kmack_phishing_urls.csv", "url", "label", {0: "benign", 1: "phishing"}),
    # Direct 4-class match, no mapping ambiguity.
    ("Malicious URL v3.csv", "url", "type",
        {"benign": "benign", "phishing": "phishing", "defacement": "defacement", "malware": "malware"}),
    ("Malicious URL Detection.csv", "url", "type",
        {"benign": "benign", "legitimate": "benign", "phishing": "phishing",
         "defacement": "defacement", "malware": "malware", "Exploit": "malware"}),
    # legitimate/phishing only.
    ("ASA_Phishing_URL_Dataset.csv", "url", "label", {"legitimate": "benign", "phishing": "phishing"}),
    ("URL dataset.csv", "url", "type", {"legitimate": "benign", "phishing": "phishing"}),
    ("StealthPhisher2025.csv", "URL", "Label", {"Legitimate": "benign", "Phishing": "phishing"}),
    # Phishing-only.
    ("Phishing URLs.csv", "url", "Type", {"Phishing": "phishing"}),
    # Binary 0/1, verified 0=benign / 1=phishing by URL sampling.
    ("Dataset.csv", "url", "label", {0: "benign", 1: "phishing"}),
    ("DomainGuard-50K.csv", "url", "label", {0: "benign", 1: "phishing"}),
    ("dataset2.csv", "URL", "Label", {0: "benign", 1: "phishing"}),
    # Binary 0/1, verified INVERTED (1=benign/legitimate / 0=phishing) by URL sampling.
    ("url_features_extracted1.csv", "URL", "ClassLabel", {1.0: "benign", 0.0: "phishing"}),
    ("PhiUSIIL_Phishing_URL_Dataset.csv", "URL", "label", {1: "benign", 0: "phishing"}),
    # Ambiguous binary "Benign vs Malicious" sources -- only the benign side
    # is usable (see module docstring); the malicious side doesn't say which
    # of our 3 malicious classes it is, so mapping it to any one of them
    # would fabricate a label. dict only maps the benign value; the
    # malicious value is simply absent, so those rows are dropped.
    ("balanced_urls.csv", "url", "label", {"benign": "benign"}),
    ("Dataset I.csv", "Websites", "Lable", {"Benign": "benign"}),
    ("Dataset II.csv", "URL", "type", {"Benign": "benign"}),
]

# abuse.ch URLhaus dump: no url/label columns to map -- every row is a
# malware-distribution URL by construction (see module docstring, priority
# tier 2). Handled separately from ADDITIONAL_SOURCES since it's a
# comment-prefixed export, not a plain CSV with a header row.
URLHAUS_PATH = DATASET_DIR / "file.csv.txt"

# PhishVN: extracted from its zip (see module docstring, priority tier 6).
PHISHVN_PATH = DATASET_DIR / "PhishVN A Time-Stamped Vietnamese URL Phishing Dat" / "data" / "dataset_url.csv"

# satyammishra0/website-url-with-industry: no label column at all (it's a
# business/industry categorization dataset, not a security one) -- every
# row is, by construction, a real registered company website, so every
# row is safely benign. Capped to a 1.5M-row sample (from 13.6M available)
# on purpose: taking the full set would push benign so far ahead that
# SMOTE's balance-to-majority-class step would need tens of millions of
# synthetic rows for every minority class, making training infeasible even
# on Kaggle.
SATYAMMISHRA_PATH = DATASET_DIR / "satyammishra_1.5M_benign_domains.csv"

# ThreatFox: authenticated bulk export (src/fetch_abusech_data.py), same
# malware-only priority tier as URLhaus -- payload-delivery/botnet-C2/
# skimming infrastructure, not phishing. Comment-prefixed CSV with quoted,
# space-padded fields (") "value"") -- skipinitialspace handles that.
THREATFOX_PATH = DATASET_DIR / "threatfox_full.csv"


def _load_additional(filename, url_col, label_col, mapping):
    path = DATASET_DIR / filename
    df = pd.read_csv(path, usecols=[url_col, label_col], low_memory=False).dropna(subset=[url_col, label_col])
    if isinstance(mapping, str):
        df["type"] = mapping
    else:
        df["type"] = df[label_col].map(mapping)
        df = df.dropna(subset=["type"])
    df["norm"] = df[url_col].astype(str).apply(normalize_url)
    return df[["norm", "type"]]


def main():
    log("Loading malicious_phish.csv (original corpus)...")
    m = pd.read_csv(DATASET_DIR / "malicious_phish.csv").dropna(subset=["url", "type"])
    m["norm"] = m["url"].astype(str).apply(normalize_url)

    log("Loading MUD_malicious_urls_2026_V2.csv (new, larger corpus)...")
    mud = pd.read_csv(DATASET_DIR / "MUD_malicious_urls_2026_V2.csv", usecols=["url", "type"]).dropna(subset=["url", "type"])
    mud["norm"] = mud["url"].astype(str).apply(normalize_url)

    log("Loading verified_online.csv (PhishTank live feed, phishing-only)...")
    v = pd.read_csv(DATASET_DIR / "verified_online.csv", usecols=["url"]).dropna(subset=["url"])
    v["type"] = "phishing"
    v["norm"] = v["url"].astype(str).apply(normalize_url)

    log("Loading merged_urls_dataset.csv (dropping out-of-taxonomy 'adult' rows)...")
    mu = pd.read_csv(DATASET_DIR / "merged_urls_dataset.csv", usecols=["url", "label"]).dropna(subset=["url", "label"])
    mu = mu[mu["label"].isin(["benign", "phish", "malware"])].rename(columns={"label": "type"})
    mu["type"] = mu["type"].replace({"phish": "phishing"})
    mu["norm"] = mu["url"].astype(str).apply(normalize_url)
    log(f"merged_urls_dataset usable (non-adult) rows: {len(mu)}")

    log("Loading URLhaus dump (file.csv.txt, malware-only, comment-prefixed export)...")
    uh = pd.read_csv(URLHAUS_PATH, comment="#", header=None,
                      names=["id", "dateadded", "url", "url_status", "last_online", "threat", "tags", "urlhaus_link", "reporter"],
                      usecols=["url"]).dropna(subset=["url"])
    uh["type"] = "malware"
    uh["norm"] = uh["url"].astype(str).apply(normalize_url)
    log(f"URLhaus usable rows: {len(uh)}")

    log("Loading PhishVN (dataset_url.csv, extracted from its zip)...")
    pv = pd.read_csv(PHISHVN_PATH, usecols=["url", "label"]).dropna(subset=["url", "label"])
    pv = pv[pv["label"].isin(["phishing", "benign"])].rename(columns={"label": "type"})
    pv["norm"] = pv["url"].astype(str).apply(normalize_url)
    log(f"PhishVN usable rows: {len(pv)}")

    log("Loading ThreatFox (url/domain IOCs, malware-only)...")
    tf = pd.read_csv(THREATFOX_PATH, comment="#", header=None, skipinitialspace=True,
                      names=["first_seen_utc", "ioc_id", "ioc_value", "ioc_type", "threat_type",
                             "fk_malware", "malware_alias", "malware_printable", "last_seen_utc",
                             "confidence_level", "is_compromised", "reference", "tags", "anonymous", "reporter"])
    tf = tf[tf["ioc_type"].isin(["url", "domain"])].dropna(subset=["ioc_value"])
    tf["type"] = "malware"
    tf["norm"] = tf["ioc_value"].astype(str).apply(normalize_url)
    log(f"ThreatFox usable rows: {len(tf)}")

    log("Loading satyammishra0 benign-domain sample (no label col -- all benign by construction)...")
    sm = pd.read_csv(SATYAMMISHRA_PATH, usecols=["website"]).dropna(subset=["website"])
    sm["type"] = "benign"
    sm["norm"] = sm["website"].astype(str).apply(normalize_url)
    log(f"satyammishra0 usable rows: {len(sm)}")

    frames = [v[["norm", "type"]], uh[["norm", "type"]], mud[["norm", "type"]], m[["norm", "type"]],
              mu[["norm", "type"]], pv[["norm", "type"]], tf[["norm", "type"]], sm[["norm", "type"]]]

    for filename, url_col, label_col, mapping in ADDITIONAL_SOURCES:
        log(f"Loading {filename}...")
        try:
            df = _load_additional(filename, url_col, label_col, mapping)
            log(f"  {filename} usable rows: {len(df)}")
            frames.append(df)
        except Exception as e:
            log(f"  SKIPPED {filename}: {e}")

    combo = pd.concat(frames, ignore_index=True)
    combo = combo[combo["type"].isin(VALID_CLASSES)]
    log(f"Total rows before dedup: {len(combo)}")

    n_labels_per_url = combo.groupby("norm")["type"].nunique()
    conflicting_urls = n_labels_per_url[n_labels_per_url > 1]
    log(f"URLs with a genuine label disagreement across sources: {len(conflicting_urls)} "
        f"({len(conflicting_urls) / n_labels_per_url.size:.2%} of {n_labels_per_url.size} unique URLs) "
        f"-- resolved by source priority order in the module docstring, not dropped.")

    combo = combo.drop_duplicates(subset="norm", keep="first").rename(columns={"norm": "url"})
    log(f"Total rows after dedup: {len(combo)}")
    log(f"Class distribution:\n{combo['type'].value_counts()}")

    out_path = DATASET_DIR / "combined_urls.csv"
    combo[["url", "type"]].to_csv(out_path, index=False)
    log(f"Saved combined dataset to {out_path}")


if __name__ == "__main__":
    main()
