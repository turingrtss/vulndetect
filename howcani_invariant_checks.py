#!/usr/bin/env python3
"""
Answering howcani's Sep 13 correction on the "89% coefficient reduction"
framing (they're right: coefficient magnitude isn't invariant under
added correlated features, so it can't be used as a decisiveness
readout in either direction). Running their 3 proposed invariant checks:

  1. ABLATION WITH LENGTH HELD IN MODEL: remove comment-derived features
     while log(document length) STAYS as a feature, report delta-AUC.
     This is invariant to what else is in the model, unlike a coefficient.

  2. HEADLINE RECOMPUTED WITH LENGTH CONTROLLED: the baseline-vs-stripped
     accuracy gap (3.6->3.95pp) with log(length) as an explicit covariate,
     AND stratified by document-length decile, pooled. If the gap
     survives within-decile (length held fixed by construction), the
     leak is real and length is just its carrier. If it collapses, the
     honest finding is "vulnerable code is shorter" -- a data fact, not
     a comment-leakage fact.

  3. SAME PAIR FOR THE PERMUTED ARM, since that's where the AUC=0.657
     came from.

Reuses the same shared-vectorizer pipeline as prior scripts.
"""
import sys
sys.path.insert(0, '/home/ubuntu/projects/vulndetect')
import numpy as np
import json
import scipy.sparse as sp
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from leakage_followup import extract_comment_blocks, length_matched_filler
from howcani_falsification_test import length_permuted_filler

print("Loading dataset...")
ds = load_dataset("lemon42-ai/Code_Vulnerability_Labeled_Dataset")
codes = [x['code'] for x in ds['train']]
labels = [x['label'] for x in ds['train']]
binary_labels = [0 if l == 'safe' else 1 for l in labels]

train_idx, test_idx = train_test_split(
    list(range(len(codes))), test_size=0.2, random_state=42, stratify=binary_labels)
y_train_arr = np.array([binary_labels[i] for i in train_idx])
y_test_arr = np.array([binary_labels[i] for i in test_idx])

X_train_baseline_codes = [codes[i] for i in train_idx]
shared_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6),
                                     max_features=50000, sublinear_tf=True)
shared_vectorizer.fit(X_train_baseline_codes)
feature_names = list(shared_vectorizer.get_feature_names_out())
marker = " # "
marker_idx = feature_names.index(marker) if marker in feature_names else None

# comment-derived feature indices: any feature that only appears in
# documents that had a comment block (approximation: features that are
# substrings of the marker/comment syntax OR co-occur near-exclusively
# with comment presence). For a clean, defensible operationalization,
# we ablate the SET of comment-region tokens by literally re-vectorizing
# text with the comment REGION masked (not just the marker token) --
# this is the direct way to "remove comment-derived features" rather
# than guessing at a token subset.
def mask_comment_region(code):
    """Keep code, replace comment content with nothing (no marker, no
    filler) -- pure code-only text, closest operationalization of
    'comment-derived features removed'."""
    code_only, _ = extract_comment_blocks(code)
    return code_only

codes_filler = length_matched_filler(codes, seed=42)
codes_perm = length_permuted_filler(codes, binary_labels, seed=42)
codes_code_only = [mask_comment_region(c) for c in codes]  # no comment region at all, no filler

def build_len_feature(idx_list, code_list):
    lens = np.array([len(code_list[i]) for i in idx_list], dtype=float)
    return np.log1p(lens).reshape(-1, 1)

def fit_with_length(text_col_train_idx, text_col_test_idx, code_list, y_train, include_text=True):
    """Fit LR with log(length) always present; include_text toggles whether
    the TF-IDF text features are included at all (ablation) or just length alone."""
    log_len_train = build_len_feature(text_col_train_idx, code_list)
    log_len_test = build_len_feature(text_col_test_idx, code_list)
    if include_text:
        Xtr_text = shared_vectorizer.transform([code_list[i] for i in text_col_train_idx])
        Xte_text = shared_vectorizer.transform([code_list[i] for i in text_col_test_idx])
        Xtr = sp.hstack([Xtr_text, sp.csr_matrix(log_len_train)]).tocsr()
        Xte = sp.hstack([Xte_text, sp.csr_matrix(log_len_test)]).tocsr()
    else:
        Xtr = sp.csr_matrix(log_len_train)
        Xte = sp.csr_matrix(log_len_test)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, y_train)
    proba_test = clf.predict_proba(Xte)[:, 1]
    return clf, proba_test

# ============================================================
# CHECK 1: Ablation with length held in the model (filler arm)
# "with comments" = filler arm (has comment-region text, replaced by
# filler tokens); "without comments" = code_only (comment region fully
# masked, nothing there at all). Both keep log(length) as a feature.
# ============================================================
print("\n--- CHECK 1: Ablation with log(length) held in the model ---")

_, proba_with = fit_with_length(train_idx, test_idx, codes_filler, y_train_arr, include_text=True)
auc_with = roc_auc_score(y_test_arr, proba_with)

_, proba_without = fit_with_length(train_idx, test_idx, codes_code_only, y_train_arr, include_text=True)
auc_without = roc_auc_score(y_test_arr, proba_without)

delta_auc = auc_with - auc_without
print(f"  AUC WITH comment-derived features (+ log-length always present): {auc_with:.4f}")
print(f"  AUC WITHOUT comment-derived features (code-only + log-length):   {auc_without:.4f}")
print(f"  Delta-AUC (comment contribution, length held fixed): {delta_auc:+.4f}")

# ============================================================
# CHECK 2: Headline gap (baseline vs stripped) recomputed with length
# controlled, AND stratified by document-length decile, pooled.
# ============================================================
print("\n--- CHECK 2: Headline gap with length controlled + decile-stratified ---")

# baseline = original code with real comments; stripped = code_only (no comments, no filler)
codes_baseline = codes  # original, with real comments intact

def fit_plain(train_i, test_i, code_list, y_train):
    Xtr = shared_vectorizer.transform([code_list[i] for i in train_i])
    Xte = shared_vectorizer.transform([code_list[i] for i in test_i])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, y_train)
    return accuracy_score(y_test_arr, clf.predict(Xte)), clf

acc_baseline_plain, _ = fit_plain(train_idx, test_idx, codes_baseline, y_train_arr)
acc_stripped_plain, _ = fit_plain(train_idx, test_idx, codes_code_only, y_train_arr)
gap_plain = acc_baseline_plain - acc_stripped_plain
print(f"  [No length control] baseline acc={acc_baseline_plain:.4f}  stripped acc={acc_stripped_plain:.4f}  gap={gap_plain*100:+.2f}pp")

# With length as explicit covariate (added feature, not stratification)
_, proba_baseline_len = fit_with_length(train_idx, test_idx, codes_baseline, y_train_arr, include_text=True)
_, proba_stripped_len = fit_with_length(train_idx, test_idx, codes_code_only, y_train_arr, include_text=True)
pred_baseline_len = (proba_baseline_len >= 0.5).astype(int)
pred_stripped_len = (proba_stripped_len >= 0.5).astype(int)
acc_baseline_len = accuracy_score(y_test_arr, pred_baseline_len)
acc_stripped_len = accuracy_score(y_test_arr, pred_stripped_len)
gap_len_covariate = acc_baseline_len - acc_stripped_len
print(f"  [Length as covariate] baseline acc={acc_baseline_len:.4f}  stripped acc={acc_stripped_len:.4f}  gap={gap_len_covariate*100:+.2f}pp")

# Decile-stratified: bin test docs by baseline document length, compute
# gap within each decile, pool (average of within-decile gaps).
test_doc_lens = np.array([len(codes_baseline[i]) for i in test_idx], dtype=float)
deciles = np.percentile(test_doc_lens, np.arange(0, 101, 10))
decile_bins = np.digitize(test_doc_lens, deciles[1:-1])  # 0..9

# refit WITHOUT length feature (plain text models), then evaluate gap within each decile
_, clf_baseline_plain = fit_plain(train_idx, test_idx, codes_baseline, y_train_arr)
_, clf_stripped_plain = fit_plain(train_idx, test_idx, codes_code_only, y_train_arr)
Xte_baseline = shared_vectorizer.transform([codes_baseline[i] for i in test_idx])
Xte_stripped = shared_vectorizer.transform([codes_code_only[i] for i in test_idx])
pred_baseline_plain_all = clf_baseline_plain.predict(Xte_baseline)
pred_stripped_plain_all = clf_stripped_plain.predict(Xte_stripped)

decile_gaps = []
decile_ns = []
for d in range(10):
    mask = decile_bins == d
    n = mask.sum()
    if n < 5:
        continue
    acc_b = accuracy_score(y_test_arr[mask], pred_baseline_plain_all[mask])
    acc_s = accuracy_score(y_test_arr[mask], pred_stripped_plain_all[mask])
    decile_gaps.append(acc_b - acc_s)
    decile_ns.append(int(n))

pooled_gap_within_decile = float(np.average(decile_gaps, weights=decile_ns))
print(f"  [Decile-stratified, pooled] mean within-decile gap: {pooled_gap_within_decile*100:+.2f}pp")
print(f"    per-decile gaps (pp): {[round(g*100,2) for g in decile_gaps]}")
print(f"    per-decile n: {decile_ns}")

# ============================================================
# CHECK 3: Same pair (ablation delta-AUC, decile-stratified gap) for
# the PERMUTED arm -- since that's where AUC=0.657 came from.
# ============================================================
print("\n--- CHECK 3: Same checks, for the PERMUTED arm ---")

_, proba_perm_with = fit_with_length(train_idx, test_idx, codes_perm, y_train_arr, include_text=True)
auc_perm_with = roc_auc_score(y_test_arr, proba_perm_with)
_, proba_perm_without = fit_with_length(train_idx, test_idx, codes_code_only, y_train_arr, include_text=True)
auc_perm_without = roc_auc_score(y_test_arr, proba_perm_without)  # same as CHECK 1 "without", reused
delta_auc_perm = auc_perm_with - auc_perm_without
print(f"  AUC WITH permuted-filler features (+ log-length always present): {auc_perm_with:.4f}")
print(f"  AUC WITHOUT (code-only + log-length, same as Check 1):            {auc_perm_without:.4f}")
print(f"  Delta-AUC (permuted-arm contribution, length held fixed): {delta_auc_perm:+.4f}")

_, clf_perm_plain = fit_plain(train_idx, test_idx, codes_perm, y_train_arr)
Xte_perm = shared_vectorizer.transform([codes_perm[i] for i in test_idx])
pred_perm_plain_all = clf_perm_plain.predict(Xte_perm)
acc_perm_plain = accuracy_score(y_test_arr, pred_perm_plain_all)
gap_perm_plain = acc_perm_plain - acc_stripped_plain
print(f"  [No length control] permuted acc={acc_perm_plain:.4f}  stripped acc={acc_stripped_plain:.4f}  gap={gap_perm_plain*100:+.2f}pp")

decile_gaps_perm = []
decile_ns_perm = []
for d in range(10):
    mask = decile_bins == d
    n = mask.sum()
    if n < 5:
        continue
    acc_p = accuracy_score(y_test_arr[mask], pred_perm_plain_all[mask])
    acc_s = accuracy_score(y_test_arr[mask], pred_stripped_plain_all[mask])
    decile_gaps_perm.append(acc_p - acc_s)
    decile_ns_perm.append(int(n))

pooled_gap_perm_within_decile = float(np.average(decile_gaps_perm, weights=decile_ns_perm))
print(f"  [Decile-stratified, pooled] mean within-decile gap: {pooled_gap_perm_within_decile*100:+.2f}pp")
print(f"    per-decile gaps (pp): {[round(g*100,2) for g in decile_gaps_perm]}")

# ============================================================
results = {
    'check1_ablation_length_held_fixed': {
        'auc_with_comment_features': round(float(auc_with), 4),
        'auc_without_comment_features': round(float(auc_without), 4),
        'delta_auc': round(float(delta_auc), 4),
    },
    'check2_headline_gap_length_controlled': {
        'gap_no_length_control_pp': round(float(gap_plain) * 100, 2),
        'gap_length_as_covariate_pp': round(float(gap_len_covariate) * 100, 2),
        'gap_decile_stratified_pooled_pp': round(pooled_gap_within_decile * 100, 2),
        'per_decile_gaps_pp': [round(g*100, 2) for g in decile_gaps],
        'per_decile_n': decile_ns,
    },
    'check3_permuted_arm_same_checks': {
        'delta_auc_length_held_fixed': round(float(delta_auc_perm), 4),
        'gap_no_length_control_pp': round(float(gap_perm_plain) * 100, 2),
        'gap_decile_stratified_pooled_pp': round(pooled_gap_perm_within_decile * 100, 2),
        'per_decile_gaps_pp': [round(g*100, 2) for g in decile_gaps_perm],
    },
}
with open('howcani_invariant_checks.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to howcani_invariant_checks.json")
