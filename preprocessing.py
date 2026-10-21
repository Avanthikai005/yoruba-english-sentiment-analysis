"""
STEP 1 - TEXT PREPROCESSING & TOKENIZATION
Sentiment Classification for English (SST-2) and Yoruba (YOSM)
Auxiliary Resource: MENYO-20k (English-Yoruba parallel corpus)

Pipeline:
  1. Parse raw datasets (handling multiline reviews in YOSM TSV files).
  2. Clean & normalize text (Unicode NFC, strip URLs, mentions, entities, preserve diacritics).
  3. Tokenize text into words and punctuation tokens.
  4. Split into train, val, and test splits (preserving benchmark splits to prevent leakage).
  5. Build vocabulary STRICTLY from train set (with optional MENYO vocabulary extension).
  6. Encode & pad sequences to MAX_LEN (64) with <pad>=0 and <unk>=1.
  7. Save processed arrays (.npz), vocab (.pkl), and dataset stats (.json).
"""

import os
import re
import csv
import json
import pickle
import unicodedata
import argparse
from urllib.request import urlretrieve
from collections import Counter

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

SEED        = 42
MAX_LEN     = 64       # max tokens per sample
MIN_FREQ    = 2        # min frequency in train set to enter vocabulary
MENYO_TOP_N = 20000    # extra Yoruba tokens pulled from MENYO-20k (for vocabulary ablation)
OUT_DIR     = os.path.join("data", "processed")
RAW_DIR     = os.path.join("data", "raw")

YOSM_REPO   = "Iyanuoluwa/YOSM"
SST2_REPO   = "stanfordnlp/sst2"
MENYO_REPO  = "UdS-LSV/menyo20k_mt"

YOSM_URLS = {
    split: f"https://huggingface.co/datasets/{YOSM_REPO}/resolve/main/{split}.tsv"
    for split in ("train", "dev", "test")
}
MENYO_URLS = {
    split: f"https://raw.githubusercontent.com/uds-lsv/menyo-20k_MT/master/data/{split}.tsv"
    for split in ("train", "dev", "test")
}

POS_TOKENS  = {"positive", "pos", "p", "1"}
NEG_TOKENS  = {"negative", "neg", "n", "0"}
NEU_TOKENS  = {"neutral", "neu", "2"}

URL_RE     = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
ENTITY_RE  = re.compile(r"&\w+;")
WS_RE      = re.compile(r"\s+")
TOKEN_RE   = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def clean_text(text: str) -> str:
    """Normalise and clean raw text while strictly preserving Yoruba diacritics.

    Yoruba tones (acute, grave) and underdots (e/ẹ, o/ọ, s/ṣ) change lexical meaning
    and syntactic role; they are preserved via Unicode NFC composition.
    """
    t = unicodedata.normalize("NFC", str(text))
    t = t.replace("\ufeff", "")                  # strip BOM
    t = URL_RE.sub(" ", t)                       # strip URLs
    t = MENTION_RE.sub(" ", t)                   # strip @mentions
    t = ENTITY_RE.sub(" ", t)                    # strip HTML entities
    t = t.replace("#", " ")                      # remove hashtag symbol, keep word
    t = t.lower()
    t = WS_RE.sub(" ", t).strip()
    return t


def tokenize(text: str):
    """Tokenize normalized text into words and punctuation."""
    return TOKEN_RE.findall(text)


def fetch_tsv(name: str, url: str) -> str:
    """Download or return cached raw TSV file from data/raw/."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.tsv")
    if not os.path.exists(path):
        print(f"  downloading {url} -> {path}")
        urlretrieve(url, path)
    return path


def parse_yosm_tsv(path: str):
    """Robust parser for YOSM TSV files handling unescaped multiline reviews.

    Raw YOSM reviews contain newline characters ('\\n') within single review fields.
    Standard csv.DictReader erroneously splits on '\\n', discarding continuation lines.
    This parser buffers lines until a row-ending sentiment label tab is encountered.
    """
    records = []
    with open(path, "r", encoding="utf-8-sig") as f:
        _header = f.readline()  # skip header: yo_review \t sentiment
        buf = []
        for line in f:
            if "\tpositive" in line or "\tnegative" in line:
                parts = line.rsplit("\t", 1)
                buf.append(parts[0])
                full_text = " ".join(buf).strip()
                sentiment_str = parts[1].strip().lower()
                label = 1 if sentiment_str == "positive" else 0
                cleaned = clean_text(full_text)
                if cleaned:
                    records.append((cleaned, label))
                buf = []
            else:
                buf.append(line.rstrip("\r\n"))
    return records


def load_yosm_splits():
    """Load official YOSM benchmark splits (train: 800, dev: 200, test: 500)."""
    splits = {}
    for split in ("train", "dev", "test"):
        path = fetch_tsv(f"yosm_{split}", YOSM_URLS[split])
        records = parse_yosm_tsv(path)
        texts = [t for t, _ in records]
        labels = np.array([y for _, y in records], dtype=np.int64)
        splits[split] = (texts, labels)
        print(f"  YOSM/{split}: {len(texts)} samples (pos: {(labels == 1).sum()}, neg: {(labels == 0).sum()})")
    return splits


def menyo_yoruba_tokens(top_n: int = MENYO_TOP_N):
    """Extract most frequent Yoruba tokens from the MENYO-20k training split."""
    tokens = []
    path = fetch_tsv("menyo_train", MENYO_URLS["train"])
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yo = row.get("Yoruba")
            if yo:
                tokens.extend(tokenize(clean_text(yo)))
    if not tokens:
        print("  [warn] No Yoruba tokens found in MENYO train split.")
        return []
    common = [tok for tok, _ in Counter(tokens).most_common(top_n)]
    print(f"  Extracted {len(common)} frequent tokens from MENYO-20k train split.")
    return common


def build_vocab(train_token_lists, min_freq=MIN_FREQ, extra_tokens=None):
    """Build vocabulary strictly from training tokens to prevent test leakage."""
    counter = Counter()
    for toks in train_token_lists:
        counter.update(toks)

    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, freq in counter.most_common():
        if freq >= min_freq:
            vocab[tok] = len(vocab)
    n_own = len(vocab)

    n_extra = 0
    if extra_tokens:
        for tok in extra_tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
                n_extra += 1

    return vocab, n_own, n_extra


def oov_rate(token_lists, vocab) -> float:
    """Calculate the Out-of-Vocabulary (OOV) token fraction."""
    total = kept = 0
    for toks in token_lists:
        for t in toks:
            total += 1
            if vocab.get(t, 1) != 1:  # not mapped to <unk>
                kept += 1
    return 1.0 - (kept / max(total, 1))


def encode_texts(texts, vocab, max_len=MAX_LEN) -> np.ndarray:
    """Encode list of strings into padded numpy array of token IDs."""
    x = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        tokens = tokenize(t)[:max_len]
        ids = [vocab.get(tok, 1) for tok in tokens]  # 1 is <unk>
        x[i, :len(ids)] = ids
    return x


def process_yosm(use_official_splits: bool = True, augment_from_menyo: bool = True):
    """Process YOSM dataset, build vocab from train, encode, and save .npz files."""
    print("\n--- Processing YOSM (Yoruba Movie Reviews) ---")
    splits_data = load_yosm_splits()

    if use_official_splits:
        print("  Using official benchmark splits: train=800, dev=200, test=500 (No leakage)")
        x_tr_txt, y_tr = splits_data["train"]
        x_va_txt, y_val = splits_data["dev"]
        x_te_txt, y_te = splits_data["test"]
    else:
        print("  Using combined stratified 70/15/15 random split")
        all_texts = splits_data["train"][0] + splits_data["dev"][0] + splits_data["test"][0]
        all_labels = np.concatenate([splits_data["train"][1], splits_data["dev"][1], splits_data["test"][1]])
        x_tr_txt, x_tmp, y_tr, y_tmp = train_test_split(
            all_texts, all_labels, test_size=0.30, random_state=SEED, stratify=all_labels)
        x_va_txt, x_te_txt, y_val, y_te = train_test_split(
            x_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    train_tokens = [tokenize(t) for t in x_tr_txt]

    extra_tokens = None
    if augment_from_menyo:
        print("  Augmenting vocabulary with MENYO-20k frequent tokens...")
        try:
            extra_tokens = menyo_yoruba_tokens(MENYO_TOP_N)
        except Exception as e:
            print(f"  [warn] MENYO-20k extraction failed ({e}); using YOSM tokens only.")
            extra_tokens = None

    vocab, n_own, n_extra = build_vocab(train_tokens, min_freq=MIN_FREQ, extra_tokens=extra_tokens)
    oov_val = oov_rate([tokenize(t) for t in x_va_txt], vocab)
    oov_test = oov_rate([tokenize(t) for t in x_te_txt], vocab)

    print(f"  Vocab: {n_own} YOSM tokens (+{n_extra} from MENYO-20k) = {len(vocab)} total")
    print(f"  OOV rate: val = {oov_val:.2%} | test = {oov_test:.2%}")

    out_dir = os.path.join(OUT_DIR, "yosm")
    os.makedirs(out_dir, exist_ok=True)

    splits_to_save = {
        "train": (x_tr_txt, y_tr),
        "val": (x_va_txt, y_val),
        "test": (x_te_txt, y_te),
    }

    for split_name, (texts, labels) in splits_to_save.items():
        encoded = encode_texts(texts, vocab, MAX_LEN)
        out_path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez(out_path, x=encoded, y=labels)
        print(f"  Saved {out_path}: x={encoded.shape}, y={labels.shape}")

    vocab_path = os.path.join(out_dir, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"  Saved {vocab_path}")

    stats = {
        "dataset": "yosm",
        "split_type": "official_benchmark" if use_official_splits else "stratified_70_15_15",
        "max_len": MAX_LEN,
        "min_freq": MIN_FREQ,
        "vocab_size": len(vocab),
        "vocab_own": n_own,
        "vocab_from_menyo": n_extra,
        "oov_rate_val": round(oov_val, 4),
        "oov_rate_test": round(oov_test, 4),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_te)),
        "train_counts": {"neg": int((y_tr == 0).sum()), "pos": int((y_tr == 1).sum())},
        "val_counts": {"neg": int((y_val == 0).sum()), "pos": int((y_val == 1).sum())},
        "test_counts": {"neg": int((y_te == 0).sum()), "pos": int((y_te == 1).sum())},
    }
    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved {stats_path}")


def verify_sst2():
    """Verify that processed SST-2 dataset exists and conforms to specifications."""
    print("\n--- Verifying SST-2 Processed Dataset ---")
    sst2_dir = os.path.join(OUT_DIR, "sst2")
    required = ["train.npz", "val.npz", "test.npz", "vocab.pkl", "stats.json"]
    all_ok = True
    for fname in required:
        p = os.path.join(sst2_dir, fname)
        if not os.path.exists(p):
            print(f"  [MISSING] {p}")
            all_ok = False
        else:
            sz = os.path.getsize(p)
            print(f"  [OK] {fname} ({sz:,} bytes)")

    if all_ok:
        with open(os.path.join(sst2_dir, "stats.json")) as f:
            stats = json.load(f)
        print(f"  SST-2 Stats: vocab={stats['vocab_size']}, train={stats['n_train']}, val={stats['n_val']}, test={stats['n_test']}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing for SST-2 and YOSM")
    parser.add_argument("--dataset", choices=["yosm", "sst2", "all"], default="yosm",
                        help="Dataset to process (default: yosm)")
    parser.add_argument("--split", choices=["official", "stratified"], default="official",
                        help="Splitting strategy for YOSM (official benchmark or stratified 70/15/15)")
    parser.add_argument("--no_menyo", action="store_true",
                        help="Do not augment YOSM vocabulary with MENYO-20k tokens")
    args = parser.parse_args()

    np.random.seed(SEED)

    if args.dataset in ("yosm", "all"):
        process_yosm(
            use_official_splits=(args.split == "official"),
            augment_from_menyo=(not args.no_menyo)
        )

    if args.dataset in ("sst2", "all"):
        verify_sst2()

    print("\nPreprocessing pipeline execution finished successfully.")


if __name__ == "__main__":
    main()