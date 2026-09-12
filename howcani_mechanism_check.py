#!/usr/bin/env python3
"""
Answering howcani's 3 requested numbers (dev.to, 2026-09-12) to settle
whether the length-matched filler arm is truly content-free or leaks a
length signal via the " # " marker token:

  1. Mean/median ORIGINAL comment length, by vulnerability label
  2. Document frequency (df) and mean TF of the " # " marker token,
     per experimental arm (baseline, stripped, filler, shuffled)
  3. Median fraction of a document that is comment text

Reuses the exact same data-prep functions as leakage_followup.py so the
numbers are directly comparable to the published results (same dataset,
same split, same fit-once vectorizer).
"""
import sys
sys.path.insert(0, '/home/ubuntu/projects/vulndetect')
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from leakage_followup import (
    strip_all_comments, extract_comment_blocks,
    cross_document_shuffle, length_matched_filler
)
import json

print("Loading dataset...")
ds = load_dataset("lemon42-ai/Code_Vulnerability_Labeled_Dataset")
codes = [x['code'] for x in ds['train']]
labels = [x['label'] for x in ds['train']]
binary_labels = [0 if l == 'safe' else 1 for l in labels]

print(f"Samples: {len(codes)}  Safe: {binary_labels.count(0)}  Vuln: {binary_labels.count(1)}")

# ---------- Q1: comment length by label ----------
comment_lengths_by_label = {0: [], 1: []}
doc_lengths = []
comment_fracs = []
for c, lbl in zip(codes, binary_labels):
    code_only, comment_text = extract_comment_blocks(c)
    comment_lengths_by_label[lbl].append(len(comment_text))
    total_len = len(c)
    doc_lengths.append(total_len)
    comment_fracs.append(len(comment_text) / total_len if total_len > 0 else 0.0)

q1 = {
    'safe_mean_comment_len': float(np.mean(comment_lengths_by_label[0])),
    'safe_median_comment_len': float(np.median(comment_lengths_by_label[0])),
    'vuln_mean_comment_len': float(np.mean(comment_lengths_by_label[1])),
    'vuln_median_comment_len': float(np.median(comment_lengths_by_label[1])),
}
print("\n--- Q1: Original comment length by label ---")
print(f"  Safe  (n={len(comment_lengths_by_label[0])}): mean={q1['safe_mean_comment_len']:.1f}  median={q1['safe_median_comment_len']:.1f}")
print(f"  Vuln  (n={len(comment_lengths_by_label[1])}): mean={q1['vuln_mean_comment_len']:.1f}  median={q1['vuln_median_comment_len']:.1f}")
diff = q1['vuln_mean_comment_len'] - q1['safe_mean_comment_len']
pct = 100 * diff / q1['safe_mean_comment_len']
print(f"  Vuln - Safe mean diff: {diff:+.1f} chars ({pct:+.1f}% relative)")

# ---------- Q3: median comment fraction of document ----------
q3 = {
    'median_comment_fraction': float(np.median(comment_fracs)),
    'mean_comment_fraction': float(np.mean(comment_fracs)),
}
print("\n--- Q3: Comment fraction of document ---")
print(f"  Median: {q3['median_comment_fraction']:.4f}  Mean: {q3['mean_comment_fraction']:.4f}")

# ---------- Q2: " # " marker df/tf per arm, using the SAME shared vectorizer ----------
X_train_baseline, X_test_baseline, y_train, y_test = train_test_split(
    codes, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels)

shared_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6),
                                     max_features=50000, sublinear_tf=True)
shared_vectorizer.fit(X_train_baseline)

feature_names = shared_vectorizer.get_feature_names_out()
marker = " # "
if marker not in feature_names:
    print(f"\nWARNING: marker {marker!r} not in fitted vocabulary — trying variants")
    candidates = [f for f in feature_names if f.strip() == '#']
    print("close matches:", candidates[:10])
marker_idx = list(feature_names).index(marker) if marker in feature_names else None

def marker_stats(test_codes, label):
    X = shared_vectorizer.transform(test_codes)
    if marker_idx is None:
        return {'arm': label, 'df': None, 'mean_tf_nonzero': None, 'note': 'marker not in vocab'}
    col = X[:, marker_idx].toarray().ravel()
    nz = col[col > 0]
    df = int((col > 0).sum())
    df_frac = df / len(col)
    mean_tf_nonzero = float(nz.mean()) if len(nz) else 0.0
    mean_tf_all = float(col.mean())
    return {
        'arm': label,
        'n_docs': len(col),
        'df_count': df,
        'df_fraction': round(df_frac, 4),
        'mean_tfidf_nonzero_docs': round(mean_tf_nonzero, 4),
        'mean_tfidf_all_docs': round(mean_tf_all, 4),
    }

arms = {}
arms['baseline'] = marker_stats(X_test_baseline, 'baseline')

codes_stripped = [strip_all_comments(c) for c in codes]
_, X_test_stripped, _, _ = train_test_split(
    codes_stripped, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels)
arms['stripped'] = marker_stats(X_test_stripped, 'stripped')

codes_filler = length_matched_filler(codes, seed=42)
_, X_test_filler, _, _ = train_test_split(
    codes_filler, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels)
arms['filler'] = marker_stats(X_test_filler, 'filler')

codes_shuffled = cross_document_shuffle(codes, seed=42)
_, X_test_shuffled, _, _ = train_test_split(
    codes_shuffled, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels)
arms['shuffled'] = marker_stats(X_test_shuffled, 'shuffled')

print("\n--- Q2: ' # ' marker token stats per arm (shared vectorizer, test split) ---")
for name, stats in arms.items():
    print(f"  {name:10s}: df={stats.get('df_count')}/{stats.get('n_docs')} ({stats.get('df_fraction')*100:.1f}%)  "
          f"mean_tfidf(nonzero docs)={stats.get('mean_tfidf_nonzero_docs')}  mean_tfidf(all)={stats.get('mean_tfidf_all_docs')}")

# ---------- Also: does marker weight correlate with original comment length? ----------
# (howcani's specific mechanism claim: marker weight tracks inverse original-comment-length)
_, X_test_idx = train_test_split(
    list(range(len(codes))), test_size=0.2, random_state=42, stratify=binary_labels)
test_comment_lens = [len(extract_comment_blocks(codes[i])[1]) for i in X_test_idx]
X_filler_test = shared_vectorizer.transform([codes_filler[i] for i in X_test_idx])
if marker_idx is not None:
    marker_col = X_filler_test[:, marker_idx].toarray().ravel()
    # correlation between marker tfidf weight (filler arm) and ORIGINAL comment length
    valid = np.array(test_comment_lens) > 0
    if valid.sum() > 10:
        corr = np.corrcoef(np.array(test_comment_lens)[valid], marker_col[valid])[0,1]
        print(f"\n--- Mechanism check: corr(original comment length, marker tfidf weight in filler arm) ---")
        print(f"  Pearson r = {corr:.4f}  (n={valid.sum()})")
        print(f"  Negative r would support howcani's 'marker weight tracks inverse comment length' hypothesis")

results = {
    'q1_comment_length_by_label': q1,
    'q3_comment_fraction': q3,
    'q2_marker_stats_by_arm': arms,
}
with open('howcani_mechanism_check.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to howcani_mechanism_check.json")
